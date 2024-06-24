
#%%

import copy
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm.auto import tqdm
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

random_seed = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    print('WARNING! Using CPU!!!')

# Load the data
folder = '../data'
with open(f'{folder}/usdc_weth_005.pickle', 'rb') as f:
    df = pickle.load(f)
df.index = pd.to_datetime(df.index)

# To avoid duplicates
potential_duplicated = df[ df.duplicated() ].block_number.unique()
sure_duplicated = list()
for block in potential_duplicated:
    temp_df = df[ df.block_number==block ]
    if (temp_df.duplicated()).sum()*2 == len(temp_df):
        sure_duplicated.append(block)
df = df[~ (df['block_number'].isin(sure_duplicated) & df.duplicated())]

# In this step, I just need the swap data
df = df[ (df.Event == 'Swap_X2Y') | (df.Event == 'Swap_Y2X') ]

# Aggregate the data in df every minute. Keep the last value of each interval
df = df.resample('1T').last()

# Split into train, validation, and test
end_date = datetime(2024, 5, 1, 00, 00, 00)

log_ret = df.price.dropna().values
log_ret = 100 * np.log(log_ret[1:] / log_ret[:-1])
log_ret = pd.Series(log_ret, index=df.price.dropna().index[1:])
x_data = log_ret[(log_ret.index >= end_date-pd.Timedelta(days=42)) &\
                 (log_ret.index < end_date)].astype(np.float64)

y_train = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=42)) &\
            (log_ret.index < end_date-pd.Timedelta(days=14))].values.astype(np.float64)).to(device)
y_train = y_train[ y_train != 0] # Remove the zeros
y_train = y_train[:len(y_train)//10]; print('Using a reduced training set')
x_train, y_train = y_train[:-1], y_train[1:]
y_val = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=14)) &\
            (log_ret.index < end_date-pd.Timedelta(days=7))].values.astype(np.float64)).to(device)
y_val = y_val[ y_val != 0] # Remove the zeros
y_val = y_val[:len(y_val)//10]; print('Using a reduced validation set')
x_val, y_val = y_val[:-1], y_val[1:]
y_test_ = torch.from_numpy(
    log_ret[(log_ret.index >= end_date-pd.Timedelta(days=7)) &\
            (log_ret.index < end_date)].values.astype(np.float64)).to(device)
y_test_ = y_test_[ y_test_ != 0 ] # Remove the zeros
x_test, y_test_ = y_test_[:-1], y_test_[1:]
z_test = torch.linspace(-torch.std(y_train), torch.std(y_train), 1000, dtype=torch.float64,
                        requires_grad=True ).unsqueeze(dim=1).to(device)
y_test = torch.concat(
    [torch.where(val <= z_test, 1., 0) for val in y_test_], dim=1
    ).T.to(device).unsqueeze(dim=2).type(torch.float64)

#%% Models Definition

import wandb

# Define the base neural network class
class BaseNN():
    '''
    Base class for neural networks.
    '''
    def __init__(self):
        '''
        Initialize the base neural network class.
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
        self.set_regularizer() # Define the regularization
    
    def set_regularizer(self):
        '''
        Set the regularization according to the parameter 'reg_type' (default: no regularization).
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
        #Two available regularization types: l1, l2, and l1_l2
        self.reg = torch.tensor(self.params['reg']).to(self.dev)
        #Eventually, apply l1 regularization
        if self.params['reg_type'] == 'l1':
            def l1_model_reg(model):
                regularization_loss = torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    regularization_loss += torch.norm(param, 1)
                return self.reg*regularization_loss
            self.regularizer = l1_model_reg
        #Eventually, apply l2 regularization
        elif self.params['reg_type'] == 'l2':
            def l2_model_reg(model):
                regularization_loss = torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    regularization_loss += torch.norm(param, 2)
                return self.reg*regularization_loss
            self.regularizer = l2_model_reg
        #Eventually, apply l1_l2 regularization
        elif self.params['reg_type'] == 'l1_l2':
            def l1_l2_model_reg(model):
                l1_loss, l2_loss = torch.tensor(0.0).to(self.dev), torch.tensor(0.0).to(self.dev)
                for param in model.parameters():
                    l1_loss += torch.norm(param, 1)
                    l2_loss += torch.norm(param, 2)
                return self.reg[0]*l1_loss + self.reg[1]*l2_loss
            self.regularizer = l1_l2_model_reg
        #Eventually, no regularization is applied
        else:
            def no_reg(model):
                return torch.tensor(0.0)
            self.regularizer = no_reg
        
    def set_optimizer(self):
        '''
        Set the optimizer according to the parameter 'optimizer' (default: Adam).
        INPUT:
            - None.
        OUTPUT:
            - None.
        '''
        #Two available optimizers: Adam, RMSprop
        temp = self.params['optimizer'].lower() if type(self.params['optimizer']) == str else None
        if temp.lower() == 'adam':
            from torch.optim import Adam
            self.opt = Adam(self.mdl.parameters(), self.params['lr'])
        elif temp.lower() == 'rmsprop':
            from torch.optim import RMSprop
            self.opt = RMSprop(self.mdl.parameters(), self.params['lr'])
        else:
            raise ValueError(f"Optimizer {temp} not recognized")
        
    def early_stopping(self, curr_loss):
        '''
        Early stopping function.
        INPUT:
            - curr_loss: float,
                current loss.
        OUTPUT:
            - output: bool,
                True if early stopping is satisfied, False otherwise.
        '''
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss #Save the best loss
            self.best_model = copy.deepcopy(self.mdl.state_dict()) #Save the best model
            self.patience = 0 #Reset the patience counter
        else:
            self.patience += 1
        #Check if I have to exit
        if self.patience > self.params['patience']:
            output = True
        else:
            output = False
        return output

    def plot_losses(self, yscale='log'):
        '''
        Plot the training loss and, eventually, the validation loss.
        INPUT:
            - yscale: str, optional
                scale of the y-axis. Default is 'log'.
        OUTPUT:
            - None.
        '''
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_theme()
        #Plot the losses
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        ax.set(yscale=yscale)
        sns.lineplot(self.train_loss, label='Train')
        if isinstance(self.val_loss, list):
            sns.lineplot(self.val_loss, label='Validation')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Losses")
        plt.legend()
        plt.show()

# Define the feedforward neural network class
class FFN(nn.Module):
    '''
    Class for Feedforward neural networks.
    '''
    def __init__(self, layers, init, activation, drop, out_act='linear', DTYPE=torch.float64):
        '''
        INPUT:
            - layers: list of int
                list such that each component is the number of neurons in the corresponding layer.
            - init: str
                the type of initialization to use for the weights. Either 'glorot_normal' or 'glorot_uniform'.
            - activation: str
                name of the activation function. Either 'tanh', 'relu', or 'sigmoid'.
            - drop: float
                dropout rate.
            - DTYPE: torch data type.
        OUTPUT:
            - None.
        '''
        super().__init__()
        # Define the layers
        self.layers = nn.ModuleList([
            nn.Linear(layers[l-1], layers[l], dtype=DTYPE) for\
                l in range(1,len(layers))
                ])
        # Initialize the weights
        self.weights_initializer(init)
        #Define activation function and dropout
        self.set_activation_function(activation, out_act) #Define activation function
        self.dropout = nn.Dropout(drop)
    
    def weights_initializer(self, init):
        '''
        Initialize the weights.
        INPUT:
            - init: str
                type of initialization to use for the weights.
        OUTPUT:
            - None.
        '''
        #Two available initializers: glorot_normal, glorot_uniform
        temp = init.lower() if type(init) == str else None
        if temp == 'glorot_normal':
            for layer in self.layers:
                nn.init.xavier_normal_(layer.weight)
        elif temp == 'glorot_uniform':
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight)
        else:
            raise ValueError(f"Initializer {init} not recognized")

    def set_activation_function(self, act, out_act):
        '''
        Set the activation function.
        INPUT:
            - activation: str
                type of activation function to use.
        OUTPUT:
            - None.
        '''
        #Four available activation functions: tanh, relu, sigmoid, linear (that is, identity)
        acts = list()
        for activation in [act, out_act]:
            if activation == 'tanh':
                acts.append( nn.Tanh() )
            elif activation == 'relu':
                acts.append( nn.ReLU() )
            elif activation == 'sigmoid':
                acts.append( nn.Sigmoid() )
            elif activation == 'softplus':
                acts.append( nn.Softplus() )
            elif activation == 'linear':
                acts.append( nn.Identity() )
            else:
                raise ValueError(f"Activation function {activation} not recognized")
        self.activation = acts[0]; self.out_act = acts[1]
        
    def forward(self, x):
        '''
        INPUT:
            - x: torch.Tensor
                input of the network; shape (batch_size, n_features).
        OUTPUT:
            - output: torch.Tensor
                output of the network; shape (batch_size, output_size).
        '''
        # Forward pass through the network
        for layer in self.layers[:-1]:
            x = self.activation(layer(x)) #Hidden layers
            x = self.dropout(x) #Dropout
        output = self.out_act(self.layers[-1](x)) #Output layer
        return output

class CRPS(nn.Module):
    '''
    Class for the CRPS loss function.
    '''
    def __init__(self, n_points=500):
        '''
        Initialize the CRPS loss function.
        INPUT:
            - quantiles: list of float
                each element is between 0 and 1 and represents a target confidence levels.
        OUTPUT:
            - None.
        '''
        super().__init__()
        self.n_points = n_points
    
    def forward(self, y_pred, y_true, indicator=False):
        '''
        INPUT:
            - y_pred: torch.Tensor
                quantile forecasts with shape (batch_size, n_series).
            - y_true: torch.Tensor
                actual values with shape (batch_size, n_series).
        OUTPUT:
            - loss: float
                mean CRPS loss.
        '''
        # Ensure to work with torch tensors
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        #Check consistency in the dimensions
        if len(y_pred.shape) == 1:
            y_pred = torch.unsqueeze(y_pred, dim=1)
        if len(y_true.shape) == 1:
            y_true = torch.unsqueeze(y_true, dim=1)
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(f'Shape[0] of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) do not match!!!')
        if y_pred.shape[1] != len(self.quantiles):
            raise ValueError(f'Shape[1] of y_pred ({y_pred.shape}) and len(quantiles) ({len(self.quantiles)}) do not match!!!')
        if y_true.shape[1] != 1:
            raise ValueError(f'Shape[1] of y_true ({y_pred.shape}) should be 1!!!')
        # Compute the pinball loss
        error = y_true - y_pred
        loss = torch.zeros(y_true.shape).to(y_true.device)
        for q, quantile in enumerate(self.quantiles):
            loss += torch.max(quantile * error[:,q:q+1], (quantile - 1) * error[:,q:q+1])
        loss = torch.mean(loss)
        return loss

# Define the NeuralCDF class
class NeuralCDF(BaseNN):
    '''
    Expected Shortfall estimation via Kratz approach with Quantile Regression Neural Network
        (QRNN) for quantile regression.
    '''
    def __init__(self, params, dev, verbose=True):
        '''
        Initialization of the K-QRNN model.
        INPUTS:
            - params: dict
                parameters of the model.
            - dev: torch.device
                indicates the device where the model will be trained.
            - verbose: bool, optional
                if True, print the training progress. Default is True.
        PARAMS:
            - optimizer: str, optional
                optimizer to use, either 'Adam' or 'RMSProp'. Default is 'Adam'.
            - reg_type: str or None, optional
                type of regularization. Either None, 'l1', 'l2', or 'l1_l2'. Default is None.
            - reg: float or list of float, optional
                regularization parameter. Not consider when reg_type=None.
                float when reg_type='l1' or 'l2'. List with two floats (l1 and l2) when
                reg_type='l1_l2'. Default is 0.
            - initializer: str, optional
                initializer for the weights. Either 'glorot_normal' or 'glorot_uniform'.
                Default is 'glorot_normal'.
            - activation: str, optional
                activation function. Either 'relu', 'sigmoid', or 'tanh'. Default is 'relu'.
            - lr: float, optional
                learning rate. Default is 0.01.
            - dropout: float, optional
                dropout rate. Default is 0.
            - batch_size: int, optional
                batch size. Default is -1, that is full batch. When
                batch_size < x_train.shape[0], mini-batch training is performed.
            - patience: int, optional
                patience for early stopping. Default is np.inf, that is no early stopping.
            - verbose: int, optional
                set after how many epochs the information on losses are printed. Default is 1.
        OUTPUTS:
            - None.
        '''
        self.set_params(params) #Set the parameters
        self.dev = dev
        self.verbose = verbose
        super().__init__()
        # Define the model and optimizer
        self.mdl = FFN(self.params['layers'], self.params['initializer'],
                       self.params['activation'], self.params['dropout'],
                       out_act=self.params['out_activation']).to(self.dev)
        self.set_optimizer() #Define the optimizer
    
    def loss_fast(self, y_pred, y_true):
        loss = torch.tensor(0, dtype=torch.float64).to(self.dev)
        for i in range(y_true.shape[0]):
            # if i==0:
            #     print('In the loss, y_pred shape:', y_pred.shape, 'y_true shape:', y_true[i].shape)
            loss += nn.functional.mse_loss(y_pred, y_true[i])
        return loss/y_true.shape[0]

    def loss_mem_save(self, y_pred, y_true, z_points):
        loss = torch.tensor(0, dtype=torch.float64).to(self.dev)
        for i in range(y_true.shape[0]):
            loss += nn.functional.mse_loss(y_pred, torch.where(y_true[i] <= z_points, 1, 0))
        return loss/y_true.shape[0]

    def set_params(self, params):
        '''
        Define the ultimate parameters dictionary by merging the parameters
            defined by the user with the default ones
        INPUT:
            - params: dict
                parameters defined by the user.
        OUTPUT:
            - None.
        '''
        self.params = {'optimizer': 'Adam', 'reg_type': None, 'out_activation':'linear',
                       'crps_std_scale': 3, 'crps_points': 500,
                       'reg': 0, 'initializer': 'glorot_normal', 'activation': 'relu',
                       'lr': 0.01, 'dropout': 0, 'batch_size':-1,
                       'patience': np.inf, 'verbose': 1, 'pdf_constr':False} #Default parameters
        self.params.update(params) #Update default parameters with those provided by the user
    
    def train_single_batch(self, z_train, y_train):
        '''
        Training with full batch.
        INPUT:
            - x_train: torch.Tensor
                model's input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's output of shape (batch_size, 1).
        OUTPUT:
            - None.
        '''
        self.mdl.train() #Set the model in training mode
        self.opt.zero_grad() # Zero the gradients
        #print(z_train.shape)
        outputs = self.mdl(z_train) # Forward pass
        #print(outputs.shape)
        #print(y_train.shape)
        loss = self.loss(outputs, y_train) + self.regularizer(self.mdl)
        if self.params['pdf_constr']:
            pdf = torch.autograd.grad(outputs, z_train,
                                      torch.ones_like(outputs), retain_graph=True)[0]
            loss += torch.mean(nn.ReLU()(-pdf))
        loss.backward()  # Backpropagation
        self.opt.step()  # Update weights
        self.train_loss.append(loss.item()) #Save the training loss
    
    def train_multi_batch(self, z_train, y_train, indices):
        '''
        Training with multiple batches.
        INPUT:
            - x_train: torch.Tensor
                model's input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's output of shape (batch_size, 1).
            - indices: torch.Tensor
                list of indices (range(batch_size)).
        OUTPUT:
            - None.
        '''
        self.mdl.train() #Set the model in training mode
        #Prepare batch training
        total_loss = 0.0
        indices = indices[torch.randperm(indices.size(0))] #Shuffle the indices
        # Training
        for i in range(0, len(indices), self.params['batch_size']):
            #Construct the batch
            batch_indices = indices[i:i+self.params['batch_size']] #Select the indices
            y_batch = y_train[batch_indices]

            self.opt.zero_grad() # Zero the gradients
            #print(z_train.shape)
            outputs = self.mdl(z_train) # Forward pass
            #print(outputs.shape)
            #print(y_batch.shape)
            loss = self.loss(outputs, y_batch) + self.regularizer(self.mdl)
            if self.params['pdf_constr']:
                pdf = torch.autograd.grad(outputs, z_train,
                                        torch.ones_like(outputs), retain_graph=True)[0]
                loss += torch.mean(nn.ReLU()(-pdf))
            loss.backward()  # Backpropagation
            self.opt.step()  # Update weights
            total_loss += loss.item()
        total_loss /= np.ceil(len(indices) / self.params['batch_size'])
        self.train_loss.append(total_loss) #Save the training loss

    def fit(self, y_train_, y_val_=None, mem_save=False, wandb_save=False):
        '''
        Fit the model. With early stopping and batch training.
        INPUT:
            - x_train: torch.Tensor
                model's train input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's train output of shape (batch_size, 1).
            - x_val: torch.Tensor, optional
                model's validation input of shape (batch_size, n_features). Default is None.
            - y_val: torch.Tensor, optional
                model's validation output of shape (batch_size, 1). Default is None.
        OUTPUT:
            - None.
        '''
        #Initialize the best model
        self.best_model = self.mdl.state_dict()
        self.best_loss = np.inf
        #Initialize the patience counter
        self.patience = 0
        #Initialize the training and validation losses
        self.train_loss = []
        if isinstance(y_val_, torch.Tensor):
            self.val_loss = []
        else:
            self.val_loss = None
        # Set the wandb environment if required:
        if wandb_save:
            wandb.init(project='unconditional_crps', config=self.params)
            wandb.watch([self.mdl], log='all')
        #Understand if I'm using single or multiple batches
        single_batch = (self.params['batch_size'] == -1) or\
            (self.params['batch_size'] >= y_train_.shape[0])
        # Eventually, create the train indices list
        if not single_batch: indices = torch.arange(y_train_.shape[0])
        # Drawn the points where evaluate the CRPS
        adj_std = torch.std(y_train_) * self.params['crps_std_scale']
        z_train = torch.linspace(
            -adj_std, adj_std, self.params['crps_points'], dtype=torch.float64,
            requires_grad = self.params['pdf_constr'] ).unsqueeze(dim=1).to(self.dev)
        if isinstance(y_val_, torch.Tensor):
            z_val = torch.distributions.uniform.Uniform(-adj_std, adj_std).sample(
                (self.params['crps_points'],)).unsqueeze(dim=1).requires_grad_(
                    requires_grad = self.params['pdf_constr']).to(self.dev)
        # If not mem_save, compute the indicator function for the train and val once at all
        if not mem_save:
            y_train = torch.concat(
                    [torch.where(val <= z_train, 1., 0) for val in y_train_],
                    dim=1).T.to(self.dev).unsqueeze(dim=2).type(torch.float64)
            if isinstance(y_val_, torch.Tensor):
                y_val = torch.concat(
                    [torch.where(val <= z_val, 1., 0).to(self.dev) for val in y_val_],
                    dim=1
                ).T.unsqueeze(dim=2)
            self.loss = self.loss_fast
        else: #Otherwise, use keep the original tensors
            y_train = y_train_
            y_val = y_val_
            self.loss = self.loss_mem_save
        # Set the verbosity if it is provided as a boolean:
        if isinstance(self.verbose, bool):
            self.verbose = int(self.verbose) - 1
        #Train the model
        if self.verbose >= 0: #If verbose is True, then use the progress bar
            from tqdm.auto import tqdm
            it_base = tqdm(range(self.params['n_epochs']), desc='Training the network') #Create the progress bar
        else: #Otherwise, use the standard iterator
            it_base = range(self.params['n_epochs']) #Create the iterator
        for epoch in it_base:
            if (epoch==0) and (self.verbose > 0):
                print_base = '{:<10}{:<15}{:<15}'
                print(print_base.format('Epoch', 'Train Loss', 'Val Loss'))
            # Training
            if single_batch:
                self.train_single_batch(z_train, y_train)
            else:
                self.train_multi_batch(z_train, y_train, indices)
            # If Validation
            if isinstance(y_val_, torch.Tensor):
                self.mdl.eval() #Set the model in evaluation mode
                if self.params['pdf_constr']:
                        # Compute the validation loss
                        val_output = self.mdl(z_val)
                        if mem_save:
                            val_loss = self.loss(val_output, y_val, z_val)
                        else:
                            val_loss = self.loss(val_output, y_val)
                        pdf = torch.autograd.grad(val_output, z_val,
                                                  torch.ones_like(val_output), retain_graph=True)[0]
                        val_loss += torch.mean(nn.ReLU()(-pdf))
                        self.val_loss.append(val_loss.item()) #Save the validation loss
                else:
                    with torch.no_grad():
                        # Compute the validation loss
                        if mem_save:
                            val_loss = self.loss(self.mdl(z_val), y_val, z_val)
                        else:
                            val_loss = self.loss(self.mdl(z_val), y_val)
                        self.val_loss.append(val_loss.item()) #Save the validation loss
                # Update best model and eventually early stopping
                if self.early_stopping(val_loss):
                    if self.verbose >= 0: print(f"Early stopping at epoch {epoch+1}")
                    break
            
            #Eventually, wandb save
            if wandb_save:
                to_wandb = dict()
                to_wandb[f'train_loss'] = self.train_loss[-1]
                to_wandb[f'val_loss'] = self.val_loss[-1]
                wandb.log(to_wandb)
            else: #Otherwise
                # Update best model and eventually early stopping
                if self.early_stopping(self.train_loss[-1]):
                    if self.verbose >= 0: print(f"Early stopping at epoch {epoch+1}")
                    break
            # Eventually, print the losses
            if self.verbose > 0:
                if (epoch+1) % self.verbose == 0:
                    self.print_losses(epoch+1)
        #After the training, load the best model and return its losses
        self.final_ops(z_train, y_train, z_val, y_val, mem_save)
        #Eventually, wandb save
        if wandb_save:
            to_wandb = dict()
            to_wandb[f'train_loss'] = self.train_loss[-1]
            to_wandb[f'val_loss'] = self.val_loss[-1]
            wandb.log(to_wandb)
        #Eventually, finish the wandb run
        if wandb_save:
            wandb.finish()

    def print_losses(self, epoch):
        print('{:<10}{:<15}{:<15}'.format(
            epoch, format(self.train_loss[-1], '.20f')[:10],
            format(self.val_loss[-1], '.20f')[:10] if not isinstance(
                self.val_loss, type(None)) else '-'))

    def final_ops(self, z_train, y_train, z_val, y_val, mem_save):
        # Recover the best model
        self.mdl.load_state_dict(self.best_model)

        # Compute the train loss of the best model
        z_train.requires_grad_(True)
        output = self.mdl(z_train)
        if mem_save:
            train_loss = self.loss(output, y_train, z_train)
        else:
            train_loss = self.loss(output, y_train)
        pdf = torch.autograd.grad(output, z_train, torch.ones_like(output))[0]
        train_loss += torch.mean(nn.ReLU()(-pdf))
        self.train_loss.append(train_loss.item())

        # Compute the val loss of the best model
        z_val.requires_grad_(True)
        val_output = self.mdl(z_val)
        if mem_save:
            val_loss = self.loss(val_output, y_val, z_val)
        else:
            val_loss = self.loss(val_output, y_val)
        pdf = torch.autograd.grad(val_output, z_val, torch.ones_like(val_output))[0]
        val_loss += torch.mean(nn.ReLU()(-pdf))
        self.val_loss.append(val_loss.item())

        # Print the loss
        print('\n')
        self.print_losses('Final')
    
    def __call__(self, x_test):
        '''
        Predict the quantile forecast and the expected shortfall.
        INPUT:
            - x_test: torch.Tensor
                input of the model; shape (batch_size, n_features).
        OUTPUT:
            - qf: ndarray
             quantile forecast of the model.
            - ef: ndarray
                expected shortfall predicted by the model.
        '''
        return self.mdl(x_test)
    
    def draw(self, n_points, z_min=-1.5, z_max=1.5, N_grid=10_000, seed=None):
        if not isinstance(seed, type(None)):
            torch.manual_seed(seed) #Set the seed
            torch.cuda.manual_seed_all(seed)
        uniform_samples = torch.rand(n_points, device=self.dev) #Sample from a uniform distribution
        
        # Create a grid of z values
        z_values = torch.linspace(z_min, z_max, N_grid, device=self.dev, dtype=torch.float64)
        z_values = z_values.unsqueeze(1)  # Make it (num_points, 1)
        
        # Evaluate the CDF for each z in the grid
        cdf_values = self.mdl(z_values).squeeze(1)  # Make it (num_points,)
        
        # Function to find the z corresponding to each uniform sample using binary search
        def find_z(cdf_value):
            low, high = 0, N_grid - 1
            while low < high:
                mid = (low + high) // 2
                if cdf_values[mid] < cdf_value:
                    low = mid + 1
                else:
                    high = mid
            return z_values[low].item()
        
        # Apply the binary search to each uniform sample
        sampled_points = torch.tensor([find_z(u) for u in uniform_samples], device=self.dev)
        
        return sampled_points

# Define the NeuralCDF class
class NeuralCondCDF(BaseNN):
    '''
    Expected Shortfall estimation via Kratz approach with Quantile Regression Neural Network
        (QRNN) for quantile regression.
    '''
    def __init__(self, params, dev, verbose=True):
        '''
        Initialization of the K-QRNN model.
        INPUTS:
            - params: dict
                parameters of the model.
            - dev: torch.device
                indicates the device where the model will be trained.
            - verbose: bool, optional
                if True, print the training progress. Default is True.
        PARAMS:
            - optimizer: str, optional
                optimizer to use, either 'Adam' or 'RMSProp'. Default is 'Adam'.
            - reg_type: str or None, optional
                type of regularization. Either None, 'l1', 'l2', or 'l1_l2'. Default is None.
            - reg: float or list of float, optional
                regularization parameter. Not consider when reg_type=None.
                float when reg_type='l1' or 'l2'. List with two floats (l1 and l2) when
                reg_type='l1_l2'. Default is 0.
            - initializer: str, optional
                initializer for the weights. Either 'glorot_normal' or 'glorot_uniform'.
                Default is 'glorot_normal'.
            - activation: str, optional
                activation function. Either 'relu', 'sigmoid', or 'tanh'. Default is 'relu'.
            - lr: float, optional
                learning rate. Default is 0.01.
            - dropout: float, optional
                dropout rate. Default is 0.
            - batch_size: int, optional
                batch size. Default is -1, that is full batch. When
                batch_size < x_train.shape[0], mini-batch training is performed.
            - patience: int, optional
                patience for early stopping. Default is np.inf, that is no early stopping.
            - verbose: int, optional
                set after how many epochs the information on losses are printed. Default is 1.
        OUTPUTS:
            - None.
        '''
        self.set_params(params) #Set the parameters
        self.dev = dev
        self.verbose = verbose
        super().__init__()
        # Define the model and optimizer
        self.mdl = FFN(self.params['layers'], self.params['initializer'],
                       self.params['activation'], self.params['dropout'],
                       out_act=self.params['out_activation']).to(self.dev)
        self.set_optimizer() #Define the optimizer
    
    def loss_fast(self, y_pred, y_true):
        loss = torch.tensor(0, dtype=torch.float64).to(self.dev)
        for i in range(y_true.shape[0]):
            # if i==0:
            #     print('In the loss, y_pred shape:', y_pred.shape, 'y_true shape:', y_true[i].shape)
            loss += nn.functional.mse_loss(y_pred, y_true[i])
        return loss/y_true.shape[0]

    def loss_mem_save(self, y_pred, y_true, z_points):
        loss = torch.tensor(0, dtype=torch.float64).to(self.dev)
        for i in range(y_true.shape[0]):
            loss += nn.functional.mse_loss(y_pred, torch.where(y_true[i] <= z_points, 1, 0))
        return loss/y_true.shape[0]

    def set_params(self, params):
        '''
        Define the ultimate parameters dictionary by merging the parameters
            defined by the user with the default ones
        INPUT:
            - params: dict
                parameters defined by the user.
        OUTPUT:
            - None.
        '''
        self.params = {'optimizer': 'Adam', 'reg_type': None, 'out_activation':'linear',
                       'crps_std_scale': 3, 'crps_points': 500,
                       'reg': 0, 'initializer': 'glorot_normal', 'activation': 'relu',
                       'lr': 0.01, 'dropout': 0, 'batch_size':-1,
                       'patience': np.inf, 'verbose': 1, 'pdf_constr':False} #Default parameters
        self.params.update(params) #Update default parameters with those provided by the user
    
    def train_single_batch(self, z_train, x_train, y_train):
        '''
        Training with full batch.
        INPUT:
            - x_train: torch.Tensor
                model's input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's output of shape (batch_size, 1).
        OUTPUT:
            - None.
        '''
        self.mdl.train() #Set the model in training mode
        self.opt.zero_grad() # Zero the gradients
        #print(z_train.shape)
        loss = 0
        for n_val, val in enumerate(x_train):
            outputs = self.mdl(
                torch.cat((z_train,
                           val.unsqueeze(dim=0).repeat(self.params['crps_points'],1)), dim=1)) # Forward pass
            loss += self.loss(outputs, y_train[n_val:n_val+1])
            if self.params['pdf_constr']:
                pdf = torch.autograd.grad(outputs, z_train,
                                        torch.ones_like(outputs), retain_graph=True)[0]
                loss += torch.mean(nn.ReLU()(-pdf))
        #print(outputs.shape)
        #print(y_train.shape)
        loss /= x_train.shape[0] #Average the loss
        loss += self.regularizer(self.mdl) #Add the regularization
        loss.backward()  # Backpropagation
        self.opt.step()  # Update weights
        self.train_loss.append(loss.item()) #Save the training loss
    
    def train_multi_batch(self, z_train, y_train, indices):
        '''
        Training with multiple batches.
        INPUT:
            - x_train: torch.Tensor
                model's input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's output of shape (batch_size, 1).
            - indices: torch.Tensor
                list of indices (range(batch_size)).
        OUTPUT:
            - None.
        '''
        self.mdl.train() #Set the model in training mode
        #Prepare batch training
        total_loss = 0.0
        indices = indices[torch.randperm(indices.size(0))] #Shuffle the indices
        # Training
        for i in range(0, len(indices), self.params['batch_size']):
            #Construct the batch
            batch_indices = indices[i:i+self.params['batch_size']] #Select the indices
            y_batch = y_train[batch_indices]

            self.opt.zero_grad() # Zero the gradients
            #print(z_train.shape)
            outputs = self.mdl(z_train) # Forward pass
            #print(outputs.shape)
            #print(y_batch.shape)
            loss = self.loss(outputs, y_batch) + self.regularizer(self.mdl)
            if self.params['pdf_constr']:
                pdf = torch.autograd.grad(outputs, z_train,
                                        torch.ones_like(outputs), retain_graph=True)[0]
                loss += torch.mean(nn.ReLU()(-pdf))
            loss.backward()  # Backpropagation
            self.opt.step()  # Update weights
            total_loss += loss.item()
        total_loss /= np.ceil(len(indices) / self.params['batch_size'])
        self.train_loss.append(total_loss) #Save the training loss

    def fit(self, x_train, y_train_, x_val=None, y_val_=None, mem_save=False, wandb_save=False):
        '''
        Fit the model. With early stopping and batch training.
        INPUT:
            - x_train: torch.Tensor
                model's train input of shape (batch_size, n_features).
            - y_train: torch.Tensor
                model's train output of shape (batch_size, 1).
            - x_val: torch.Tensor, optional
                model's validation input of shape (batch_size, n_features). Default is None.
            - y_val: torch.Tensor, optional
                model's validation output of shape (batch_size, 1). Default is None.
        OUTPUT:
            - None.
        '''
        #Initialize the best model
        self.best_model = self.mdl.state_dict()
        self.best_loss = np.inf
        #Initialize the patience counter
        self.patience = 0
        #Initialize the training and validation losses
        self.train_loss = []
        if isinstance(y_val_, torch.Tensor):
            self.val_loss = []
        else:
            self.val_loss = None
        # Set the wandb environment if required:
        if wandb_save:
            wandb.init(project='conditional_tt1_crps', config=self.params)
            wandb.watch([self.mdl], log='all')
        #Understand if I'm using single or multiple batches
        single_batch = (self.params['batch_size'] == -1) or\
            (self.params['batch_size'] >= y_train_.shape[0])
        # Eventually, create the train indices list
        if not single_batch: indices = torch.arange(y_train_.shape[0])
        # Drawn the points where evaluate the CRPS
        adj_std = torch.std(y_train_) * self.params['crps_std_scale']
        z_train = torch.linspace(
            -adj_std, adj_std, self.params['crps_points'], dtype=torch.float64,
            requires_grad = self.params['pdf_constr'] ).unsqueeze(dim=1).to(self.dev)
        if isinstance(y_val_, torch.Tensor):
            z_val = torch.distributions.uniform.Uniform(-adj_std, adj_std).sample(
                (self.params['crps_points'],)).unsqueeze(dim=1).requires_grad_(
                    requires_grad = self.params['pdf_constr']).to(self.dev)
        # If not mem_save, compute the indicator function for the train and val once at all
        if not mem_save:
            y_train = torch.concat(
                    [torch.where(val <= z_train, 1., 0) for val in y_train_],
                    dim=1).T.to(self.dev).unsqueeze(dim=2).type(torch.float64)
            if isinstance(y_val_, torch.Tensor):
                y_val = torch.concat(
                    [torch.where(val <= z_val, 1., 0).to(self.dev) for val in y_val_],
                    dim=1
                ).T.unsqueeze(dim=2)
            self.loss = self.loss_fast
        else: #Otherwise, use keep the original tensors
            y_train = y_train_
            y_val = y_val_
            self.loss = self.loss_mem_save
        # Set the verbosity if it is provided as a boolean:
        if isinstance(self.verbose, bool):
            self.verbose = int(self.verbose) - 1
        #Train the model
        if self.verbose >= 0: #If verbose is True, then use the progress bar
            from tqdm.auto import tqdm
            it_base = tqdm(range(self.params['n_epochs']), desc='Training the network') #Create the progress bar
        else: #Otherwise, use the standard iterator
            it_base = range(self.params['n_epochs']) #Create the iterator
        for epoch in it_base:
            if (epoch==0) and (self.verbose > 0):
                print_base = '{:<10}{:<15}{:<15}'
                print(print_base.format('Epoch', 'Train Loss', 'Val Loss'))
            # Training
            if single_batch:
                self.train_single_batch(z_train, x_train, y_train)
            else:
                self.train_multi_batch(z_train, y_train, indices)
            # If Validation
            if isinstance(y_val_, torch.Tensor):
                self.mdl.eval() #Set the model in evaluation mode
                if self.params['pdf_constr']:
                    # Compute the validation loss
                    val_loss = 0
                    for n_val, x_value in enumerate(x_val):
                        val_output = self.mdl(
                            torch.cat((z_val,
                                    x_value.unsqueeze(dim=0).repeat(self.params['crps_points'],1)), dim=1)) # Forward pass
                        val_loss += self.loss(val_output, y_val[n_val:n_val+1])
                        pdf = torch.autograd.grad(val_output, z_val,
                                                torch.ones_like(val_output), retain_graph=True)[0]
                        val_loss += torch.mean(nn.ReLU()(-pdf))
                    val_loss /= x_val.shape[0] #Average the loss
                    self.val_loss.append(val_loss.item()) #Save the validation loss
                else:
                    with torch.no_grad():
                        # Compute the validation loss
                        if mem_save:
                            val_loss = self.loss(self.mdl(z_val), y_val, z_val)
                        else:
                            val_loss = 0
                            for n_val, x_value in enumerate(x_val):
                                val_output = self.mdl(
                                    torch.cat((z_val,
                                            x_value.unsqueeze(dim=0).repeat(self.params['crps_points'],1)), dim=1)) # Forward pass
                                val_loss += self.loss(val_output, y_val[n_val:n_val+1])
                            val_loss /= x_val.shape[0] #Average the loss
                        self.val_loss.append(val_loss.item()) #Save the validation loss
                # Update best model and eventually early stopping
                if self.early_stopping(val_loss):
                    if self.verbose >= 0: print(f"Early stopping at epoch {epoch+1}")
                    break
            
            #Eventually, wandb save
            if wandb_save:
                to_wandb = dict()
                to_wandb[f'train_loss'] = self.train_loss[-1]
                to_wandb[f'val_loss'] = self.val_loss[-1]
                wandb.log(to_wandb)
            else: #Otherwise
                # Update best model and eventually early stopping
                if self.early_stopping(self.train_loss[-1]):
                    if self.verbose >= 0: print(f"Early stopping at epoch {epoch+1}")
                    break
            # Eventually, print the losses
            if self.verbose > 0:
                if (epoch+1) % self.verbose == 0:
                    self.print_losses(epoch+1)
        #After the training, load the best model and return its losses
        self.final_ops(z_train, x_train, y_train, z_val, x_val, y_val, mem_save)
        #Eventually, wandb save
        if wandb_save:
            to_wandb = dict()
            to_wandb[f'train_loss'] = self.train_loss[-1]
            to_wandb[f'val_loss'] = self.val_loss[-1]
            wandb.log(to_wandb)
        #Eventually, finish the wandb run
        if wandb_save:
            wandb.finish()

    def print_losses(self, epoch):
        print('{:<10}{:<15}{:<15}'.format(
            epoch, format(self.train_loss[-1], '.20f')[:10],
            format(self.val_loss[-1], '.20f')[:10] if not isinstance(
                self.val_loss, type(None)) else '-'))

    def final_ops(self, z_train, x_train, y_train, z_val, x_val, y_val, mem_save):
        # Recover the best model
        self.mdl.load_state_dict(self.best_model)

        # Compute the train loss of the best model
        z_train.requires_grad_(True)
        if mem_save:
            train_loss = self.loss(output, y_train, z_train)
        else:
            train_loss = 0
            for n_val, x_value in enumerate(x_train):
                output = self.mdl(
                    torch.cat((z_train,
                            x_value.unsqueeze(dim=0).repeat(self.params['crps_points'],1)), dim=1)) # Forward pass
                train_loss += self.loss(output, y_train[n_val:n_val+1])
                pdf = torch.autograd.grad(output, z_train,
                                        torch.ones_like(output), retain_graph=True)[0]
                train_loss += torch.mean(nn.ReLU()(-pdf))
            train_loss /= x_train.shape[0] #Average the loss
        self.train_loss.append(train_loss.item())

        # Compute the val loss of the best model
        z_val.requires_grad_(True)
        if mem_save:
            val_loss = self.loss(val_output, y_val, z_val)
        else:
            val_loss = 0
            for n_val, x_value in enumerate(x_val):
                val_output = self.mdl(
                    torch.cat((z_val,
                            x_value.unsqueeze(dim=0).repeat(self.params['crps_points'],1)), dim=1)) # Forward pass
                val_loss += self.loss(val_output, y_val[n_val:n_val+1])
                pdf = torch.autograd.grad(val_output, z_val,
                                        torch.ones_like(val_output), retain_graph=True)[0]
                val_loss += torch.mean(nn.ReLU()(-pdf))
            val_loss /= x_val.shape[0] #Average the loss
        self.val_loss.append(val_loss.item())

        # Print the loss
        print('\n')
        self.print_losses('Final')
    
    def __call__(self, z_test, x_test):
        '''
        Predict the quantile forecast and the expected shortfall.
        INPUT:
            - x_test: torch.Tensor
                input of the model; shape (batch_size, n_features).
        OUTPUT:
            - qf: ndarray
             quantile forecast of the model.
            - ef: ndarray
                expected shortfall predicted by the model.
        '''
        return self.mdl(torch.cat((z_test,
                                    x_test), dim=1))
    
    def draw(self, x0, num_points, z_min=-10.0, z_max=10.0, N_grid=20_000, seed=None):
        """
        Generate a path of points from a conditional distribution represented by a neural network mdl.
        
        Args:
            mdl (nn.Module): Neural network representing the conditional CDF.
            x0 (float): Starting point x_0.
            num_points (int): Number of points to generate in the path.
            z_min (float): Minimum value of z to consider in the search.
            z_max (float): Maximum value of z to consider in the search.
            N_grid (int): Number of points to use in the search grid.
            seed (int): Seed for the random number generator.
            
        Returns:
            torch.Tensor: Tensor of generated points.
        """
        if not isinstance(seed, type(None)):
            torch.manual_seed(seed) #Set the seed
            torch.cuda.manual_seed_all(seed)
        
        # Create a grid of z values
        z_values = torch.linspace(z_min, z_max, N_grid, device=self.dev, dtype=torch.float64)
        z_values = z_values.unsqueeze(1)  # Make it (N_grid, 1)
        
        # Initialize the path with the starting point
        path = [x0]
        
        for _ in range(1, num_points):
            # Get the previous point
            x_prev = path[-1]
            
            # Generate a uniform random sample in [0, 1]
            uniform_sample = torch.rand(1, device=self.dev).item()
            
            # Create the input tensor for the model (z_values, x_prev)
            inputs = torch.cat([z_values, torch.full((N_grid, 1), x_prev, device=self.dev)], dim=1)
            
            # Evaluate the CDF for each z in the grid given x_prev
            with torch.no_grad():
                cdf_values = self.mdl(inputs).squeeze(1)  # Make it (N_grid,)
            
            # Function to find the z corresponding to the uniform sample using binary search
            def find_z(cdf_value):
                low, high = 0, N_grid - 1
                while low < high:
                    mid = (low + high) // 2
                    if cdf_values[mid] < cdf_value:
                        low = mid + 1
                    else:
                        high = mid
                return z_values[low].item()
            
            # Find the z corresponding to the uniform sample
            next_point = find_z(uniform_sample)
            
            # Append the generated point to the path
            path.append(next_point)
        
        sampled_points = torch.tensor(path, device=self.dev)
        return sampled_points

#%% Unconditional CDF - HP Tuning

import random
# Define the hyperparameters grid
param_grid = {'activation': ['relu', 'tanh', 'sigmoid', 'softplus'],
              'out_activation': ['linear', 'sigmoid'],
            'reg_type': ['l1', 'l2', 'l1_l2', None],
            'reg': [[i, j] for i in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6] for j in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]],
            'lr': [1e-2, 5e-3, 3e-3, 1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5],
            'initializer': ['glorot_normal', 'glorot_uniform'],
            'optimizer': ['adam', 'rmsprop'],
            'dropout': [0, 0.2, 0.4, 0.5],
            'n_layers': [2, 3, 4, 5],
            'layer_size': [10, 20, 50],
            'pdf_constr': [True, False],
            'n_epochs': [3_000],
            'patience': [20],
            'batch_size': [-1],
            'crps_points': [150]}

best_seed, best_loss = None, np.inf
for rgs_seed in tqdm(range(90), desc=f'Randomized grid search'):
    # Sample from the grid
    random.seed(rgs_seed)
    temp_par = dict()
    for key in param_grid.keys():
        temp_par[key] = random.choice(param_grid[key])
    #Set the regularization
    if temp_par['reg_type'] != 'l1_l2':
        temp_par['reg'] = temp_par['reg'][0]
    # Set the layers
    temp_par['layers'] = [1] + [temp_par.pop('layer_size')]*temp_par.pop('n_layers') + [1]

    # Create and train the model
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    print(rgs_seed)
    print(temp_par)
    mdl = NeuralCDF(temp_par, device, verbose=False)
    mdl.fit(y_train, y_val, wandb_save=True)

    # Final evaluation
    output = mdl.mdl(z_test)
    train_loss = mdl.loss(output, y_test)
    pdf = torch.autograd.grad(output, z_test, torch.ones_like(output))[0]
    train_loss += torch.mean(nn.ReLU()(-pdf))
    print(train_loss.item())
    if train_loss < best_loss:
        best_seed, best_loss = rgs_seed, train_loss.item()
        print(f'Found new point!!!\nBest seed: {best_seed}, Best loss: {best_loss}')
    

    # Plot pdf
    F_z_batch = mdl(z_test)
    f_z_batch = torch.autograd.grad(F_z_batch, z_test, grad_outputs=torch.ones_like(F_z_batch))[0]
    # sort F_z_batch
    F_z_batch, _ = torch.sort(F_z_batch)
    f_z_batch, _ = torch.sort(f_z_batch)
    # convert to numpy both z_batch and F_z_batch
    F_z_batch = F_z_batch.cpu().detach().numpy()
    f_z_batch = f_z_batch.cpu().detach().numpy()
    z_batch = z_test.cpu().detach().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # plot the CDF
    sns.lineplot(x=z_batch.flatten(), y=F_z_batch.flatten(), ax=ax[0])
    # plot the real CDF
    y = norm.cdf(z_batch,
                loc=torch.mean(torch.concatenate([y_train, y_val])).item(),
                scale=torch.std(torch.concatenate([y_train, y_val])).item())
    sns.lineplot(x=z_batch.flatten(), y=y.flatten(), ax=ax[0])
    plt.figure()
    sns.lineplot(x=z_batch.flatten(), y=f_z_batch.flatten(), ax=ax[1])
    # plot the real CDF
    y = norm.pdf(z_batch,
                loc=torch.mean(torch.concatenate([y_train, y_val])).item(),
                scale=torch.std(torch.concatenate([y_train, y_val])).item())
    sns.lineplot(x=z_batch.flatten(), y=y.flatten(), ax=ax[1])
    plt.show()

    print('\n')

# Store and print the most promising hp combionations
#   to see if there is some recurrent pattern
note_runs = []
for rgs_seed in note_runs:
    # Sample from the grid
    random.seed(rgs_seed)
    temp_par = dict()
    for key in param_grid.keys():
        temp_par[key] = random.choice(param_grid[key])
    #Set the regularization
    if temp_par['reg_type'] != 'l1_l2':
        temp_par['reg'] = temp_par['reg'][0]
    # Set the layers
    temp_par['layers'] = [1] + [temp_par.pop('layer_size')]*temp_par.pop('n_layers') + [1]
    print(temp_par)

#%% Unconditional CDF

params = {'activation': 'tanh', 'out_activation': 'sigmoid', 'reg_type': None,
          'reg': 1e-02, 'lr': 1e-03, 'initializer': 'glorot_normal',
          'optimizer': 'adam', 'dropout': 0.2, 'pdf_constr': True,
          'n_epochs': 3_000, 'patience': 20, 'batch_size': -1, 'crps_points': 150,
          'layers': [1, 50, 50, 50, 1], 'crps_std_scale': 3}


torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

mdl_un = NeuralCDF(params, device, verbose=30)
mdl_un.fit(y_train, y_val)
mdl_un.plot_losses()

# Final evaluation
output = mdl_un.mdl(z_test)
train_loss = mdl_un.loss(output, y_test)
pdf = torch.autograd.grad(output, z_test, torch.ones_like(output))[0]
train_loss += torch.mean(nn.ReLU()(-pdf))
print(train_loss.item())

# Plot CDF and PDF
z_plot = torch.linspace(-torch.std(y_train)*3, torch.std(y_train)*3, 1000, dtype=torch.float64,
                        requires_grad=True ).unsqueeze(dim=1).to(device)
# Compute the CDF
F_z_batch = mdl_un(z_plot)
f_z_batch = torch.autograd.grad(F_z_batch, z_plot, grad_outputs=torch.ones_like(F_z_batch))[0]
# sort F_z_batch
F_z_batch, _ = torch.sort(F_z_batch)
f_z_batch, _ = torch.sort(f_z_batch)
# convert to numpy both z_batch and F_z_batch
F_z_batch = F_z_batch.cpu().detach().numpy()
f_z_batch = f_z_batch.cpu().detach().numpy()
z_batch = z_plot.cpu().detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# plot the CDF
sub_p = 0
sns.lineplot(x=z_batch.flatten(), y=F_z_batch.flatten(), ax=ax[sub_p], label='Estimated CDF')
# plot the real CDF
y = norm.cdf(z_batch,
             loc=torch.mean(torch.concatenate([y_train, y_val])).item(),
             scale=torch.std(torch.concatenate([y_train, y_val])).item())
sns.lineplot(x=z_batch.flatten(), y=y.flatten(), ax=ax[sub_p], label='Normal CDF')
ax[sub_p].legend()

sub_p = 1
sns.lineplot(x=z_batch.flatten(), y=f_z_batch.flatten(), ax=ax[sub_p], label='Estimated PDF')
# plot the real CDF
y = norm.pdf(z_batch,
             loc=torch.mean(torch.concatenate([y_train, y_val])).item(),
             scale=torch.std(torch.concatenate([y_train, y_val])).item())
sns.lineplot(x=z_batch.flatten(), y=y.flatten(), ax=ax[sub_p], label='Normal PDF')
ax[sub_p].legend()
plt.show()

#--------------------------------------------- q-Gaussian diistribution
# Fit the q-Gaussian distribution on the real data
fitted_values_y = qGaussian.fit(y_train.cpu().detach().numpy(), n_it=200)
fitted_q_y, fitted_mu_y, fitted_sigma_y =\
    fitted_values_y['q'], fitted_values_y['mu'], fitted_values_y['sigma']
print(f"Fitted q: {fitted_q_y}, Fitted mu: {fitted_mu_y}, Fitted sigma: {fitted_sigma_y}")

# Fitted q: 1.5810021301905273, Fitted mu: -0.0005259437239173324, Fitted sigma: 0.048677821154237945

uncond_gen = mdl_un.draw(mdl_un.mdl, device, 10_000,
                         
                         z_max=np.max(y_train.cpu().detach().numpy()),
                         seed=random_seed)

uncond_gen = draw(mdl_un.mdl, device, len(y_train),
                  z_min=torch.min(y_train).item(),
                  z_max=torch.max(y_train).item(),
                  seed=random_seed).cpu().detach().numpy()

# Plot the trajectory
y_temp = uncond_gen/100
y_temp = np.exp( y_temp.cumsum() ) * df.price[0]
plt.plot(y_temp)

# Comparison between stats
print('{:<18}{:<2}{:<15}{:<15}'.format('Statistic', '|', 'True', 'Generated'))
print('{:<50}'.format('-'*50))
print('{:<18}{:<2}{:<15}{:<15}'.format('Min', '|', round(torch.min(y_train).item(), 5), round(np.min(uncond_gen), 5)))
print('{:<18}{:<2}{:<15}{:<15}'.format('Quantile 0.25', '|', round(torch.quantile(y_train, 0.25).item(), 5), round(np.quantile(uncond_gen, 0.25), 5)))
print('{:<18}{:<2}{:<15}{:<15}'.format('Mean', '|', round(torch.mean(y_train).item(), 5), round(np.mean(uncond_gen), 5)))
print('{:<18}{:<2}{:<15}{:<15}'.format('Median', '|', round(torch.median(y_train).item(), 5), round(np.median(uncond_gen), 5)))
print('{:<18}{:<2}{:<15}{:<15}'.format('Quantile 0.75', '|', round(torch.quantile(y_train, 0.75).item(), 5), round(np.quantile(uncond_gen, 0.75), 5)))
print('{:<18}{:<2}{:<15}{:<15}'.format('Max', '|', round(torch.max(y_train).item(), 5), round(np.max(uncond_gen), 5)))

# Fit the q-Gaussian distribution
import qGaussian
fitted_values = qGaussian.fit(uncond_gen, n_it=200)
fitted_q, fitted_mu, fitted_sigma =\
    fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
print(f"Fitted q: {fitted_q}, Fitted mu: {fitted_mu}, Fitted sigma: {fitted_sigma}")

# Fitted q: 1.63412601637673, Fitted mu: -2.0687180083272913e-05, Fitted sigma: 0.048127772580076694

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.histplot(uncond_gen, ax=ax, stat='probability', label='Data')

num_bins = len(ax.patches) # Number of bins
temp = ax.patches
hist_values = np.histogram(uncond_gen, bins=num_bins, density=True)[0] #Obtain the histogram values
hist_values /= np.sum(hist_values) # Normalize the values
pdf = qGaussian.pdf(np.linspace(np.min(uncond_gen), np.max(uncond_gen), 10000), fitted_q, mu=fitted_mu, sigma=fitted_sigma)

sns.lineplot(x=np.linspace(np.min(uncond_gen), np.max(uncond_gen), 10000),
             y=pdf*np.max(hist_values)/np.max(pdf),
            ax=ax, color=sns.color_palette()[1], label='Scaled qGaussian PDF')

ax.legend()
plt.show()

#%% Conditional x_t|x_{t-1} CDF - HP Tuning

import random
# Define the hyperparameters grid
param_grid = {'activation': ['relu', 'tanh', 'sigmoid', 'softplus'],
              'out_activation': ['linear', 'sigmoid'],
            'reg_type': ['l1', 'l2', 'l1_l2', None],
            'reg': [[i, j] for i in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6] for j in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]],
            'lr': [1e-2, 5e-3, 3e-3, 1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5],
            'initializer': ['glorot_normal', 'glorot_uniform'],
            'optimizer': ['adam', 'rmsprop'],
            'dropout': [0, 0.2, 0.4, 0.5],
            'n_layers': [2, 3, 4, 5],
            'layer_size': [10, 20, 50],
            'pdf_constr': [True, False],
            'n_epochs': [3_000],
            'patience': [20],
            'batch_size': [-1],
            'crps_points': [100]}

best_seed, best_loss = None, np.inf
for rgs_seed in tqdm(range(90), desc=f'Randomized grid search'):
    # Sample from the grid
    random.seed(rgs_seed)
    temp_par = dict()
    for key in param_grid.keys():
        temp_par[key] = random.choice(param_grid[key])
    #Set the regularization
    if temp_par['reg_type'] != 'l1_l2':
        temp_par['reg'] = temp_par['reg'][0]
    # Set the layers
    temp_par['layers'] = [2] + [temp_par.pop('layer_size')]*temp_par.pop('n_layers') + [1]

    # Create and train the model
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    print(rgs_seed)
    print(temp_par)
    mdl = NeuralCondCDF(temp_par, device, verbose=False)
    mdl.fit(x_train, y_train, x_val, y_val, wandb_save=True)
    # mdl = NeuralCondCDF(temp_par, device, verbose=10)
    # mdl.fit(x_train, y_train, x_val, y_val, wandb_save=False)

    # Final evaluation
    output = mdl.mdl(z_test)
    train_loss = mdl.loss(output, y_test)
    pdf = torch.autograd.grad(output, z_test, torch.ones_like(output))[0]
    train_loss += torch.mean(nn.ReLU()(-pdf))
    print(train_loss.item())
    if train_loss < best_loss:
        best_seed, best_loss = rgs_seed, train_loss.item()
        print(f'Found new point!!!\nBest seed: {best_seed}, Best loss: {best_loss}')
    
    # Plot pdf
    F_z_batch = mdl(z_test)
    f_z_batch = torch.autograd.grad(F_z_batch, z_test, grad_outputs=torch.ones_like(F_z_batch))[0]
    # sort F_z_batch
    F_z_batch, _ = torch.sort(F_z_batch)
    f_z_batch, _ = torch.sort(f_z_batch)
    # convert to numpy both z_batch and F_z_batch
    F_z_batch = F_z_batch.cpu().detach().numpy()
    f_z_batch = f_z_batch.cpu().detach().numpy()
    z_batch = z_test.cpu().detach().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # plot the CDF
    sns.lineplot(x=z_batch.flatten(), y=F_z_batch.flatten(), ax=ax[0])
    # plot the real CDF
    y = norm.cdf(z_batch,
                loc=torch.mean(torch.concatenate([y_train, y_val])).item(),
                scale=torch.std(torch.concatenate([y_train, y_val])).item())
    sns.lineplot(x=z_batch.flatten(), y=y.flatten(), ax=ax[0])
    plt.figure()
    sns.lineplot(x=z_batch.flatten(), y=f_z_batch.flatten(), ax=ax[1])
    # plot the real CDF
    y = norm.pdf(z_batch,
                loc=torch.mean(torch.concatenate([y_train, y_val])).item(),
                scale=torch.std(torch.concatenate([y_train, y_val])).item())
    sns.lineplot(x=z_batch.flatten(), y=y.flatten(), ax=ax[1])
    plt.show()

    print('\n')

# Store and print the most promising hp combionations
#   to see if there is some recurrent pattern
note_runs = []
for rgs_seed in note_runs:
    # Sample from the grid
    random.seed(rgs_seed)
    temp_par = dict()
    for key in param_grid.keys():
        temp_par[key] = random.choice(param_grid[key])
    #Set the regularization
    if temp_par['reg_type'] != 'l1_l2':
        temp_par['reg'] = temp_par['reg'][0]
    # Set the layers
    temp_par['layers'] = [1] + [temp_par.pop('layer_size')]*temp_par.pop('n_layers') + [1]
    print(temp_par)

#%% Conditional x_t|x_{t-1} CDF

params = {'activation': 'tanh', 'out_activation': 'sigmoid', 'reg_type': 'l1_l2',
          'reg': [1e-05, 1e-06], 'lr': 3e-03, 'initializer': 'glorot_normal',
          'optimizer': 'adam', 'dropout': 0.5, 'pdf_constr': False,
          'n_epochs': 3_000, 'patience': 20, 'batch_size': -1, 'crps_points': 100,
          'layers': [2, 10, 10, 10, 1], 'crps_std_scale': 3}

torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

mdl = NeuralCondCDF(params, device, verbose=5)
mdl.fit(x_train, y_train, x_val, y_val, wandb_save=False)
mdl.plot_losses()

# Final evaluation
mdl.mdl.eval()
test_loss = 0
for n_val, x_value in enumerate(x_val):
    val_output = mdl.mdl(
        torch.cat((z_test,
                x_value.unsqueeze(dim=0).repeat(1000,1)), dim=1)) # Forward pass
    test_loss += mdl.loss(val_output, y_val[n_val:n_val+1])
test_loss /= x_val.shape[0] #Average the loss
print(test_loss.item()) #Save the validation loss

#-------------------------------------- Plot an example
x_value = x_val[-1]
print('Conditioning on S_{t-1}=', x_value.item())
z_batch = torch.linspace(-3*torch.std(y_train), 3*torch.std(y_train), 1000, dtype=torch.float64,
                        requires_grad=True ).unsqueeze(dim=1).to(device)
F_z_batch = mdl.mdl(
    torch.cat((z_batch, x_value.unsqueeze(dim=0).repeat(1000,1)), dim=1))
f_z_batch = torch.autograd.grad(F_z_batch, z_batch,
                        torch.ones_like(F_z_batch))[0]
# sort F_z_batch
F_z_batch, _ = torch.sort(F_z_batch)
f_z_batch, _ = torch.sort(f_z_batch)
# convert to numpy both z_batch and F_z_batch
F_z_batch = F_z_batch.cpu().detach().numpy()
f_z_batch = f_z_batch.cpu().detach().numpy()
z_batch = z_batch.cpu().detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# plot the CDF
sub_p = 0
sns.lineplot(x=z_batch.flatten(), y=F_z_batch.flatten(), ax=ax[sub_p], label='Estimated CDF')
# plot the real CDF
y = norm.cdf(z_batch,
             loc=0,
             scale=torch.std(torch.concatenate([y_train, y_val])).item())
sns.lineplot(x=z_batch.flatten(), y=y.flatten(), ax=ax[sub_p], label='Normal CDF')
ax[sub_p].legend()

sub_p = 1
sns.lineplot(x=z_batch.flatten(), y=f_z_batch.flatten(), ax=ax[sub_p], label='Estimated PDF')
# plot the real CDF
y = norm.pdf(z_batch,
             loc=0,
             scale=torch.std(torch.concatenate([y_train, y_val])).item())
sns.lineplot(x=z_batch.flatten(), y=y.flatten(), ax=ax[sub_p], label='Normal PDF')
ax[sub_p].legend()
plt.show()

#-------------------------------------- Comparing two different conditioning events
x_value1 = x_train[0]
x_value2 = x_train[1]
z_batch = torch.linspace(-3*torch.std(y_train), 3*torch.std(y_train), 1000, dtype=torch.float64,
                        requires_grad=True ).unsqueeze(dim=1).to(device)
F_z_batch1 = mdl.mdl(
    torch.cat((z_batch, x_value1.unsqueeze(dim=0).repeat(1000,1)), dim=1))
f_z_batch1 = torch.autograd.grad(F_z_batch1, z_batch,
                        torch.ones_like(F_z_batch1))[0]
F_z_batch2 = mdl.mdl(
    torch.cat((z_batch, x_value2.unsqueeze(dim=0).repeat(1000,1)), dim=1))
f_z_batch2 = torch.autograd.grad(F_z_batch2, z_batch,
                        torch.ones_like(F_z_batch2))[0]
# sort F_z_batch
F_z_batch1, _ = torch.sort(F_z_batch1)
f_z_batch1, _ = torch.sort(f_z_batch1)
F_z_batch2, _ = torch.sort(F_z_batch2)
f_z_batch2, _ = torch.sort(f_z_batch2)
# convert to numpy both z_batch and F_z_batch
F_z_batch1 = F_z_batch1.cpu().detach().numpy()
f_z_batch1 = f_z_batch1.cpu().detach().numpy()
F_z_batch2 = F_z_batch2.cpu().detach().numpy()
f_z_batch2 = f_z_batch2.cpu().detach().numpy()
z_batch = z_batch.cpu().detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# plot the CDF
sub_p = 0
sns.lineplot(x=z_batch.flatten(), y=F_z_batch.flatten(), ax=ax[sub_p])
sns.lineplot(x=z_batch.flatten(), y=F_z_batch1.flatten(), ax=ax[sub_p])
sns.lineplot(x=z_batch.flatten(), y=F_z_batch2.flatten(), ax=ax[sub_p])
# plot the real CDF
y = norm.cdf(z_batch,
             loc=0,
             scale=torch.std(torch.concatenate([y_train, y_val])).item())
ax[sub_p].set_title('Cumulative Density Function')
sns.lineplot(x=z_batch.flatten(), y=y.flatten(), ax=ax[sub_p])

sub_p = 1
sns.lineplot(x=z_batch.flatten(), y=f_z_batch.flatten(), ax=ax[sub_p], label='Unconditioned')
sns.lineplot(x=z_batch.flatten(), y=f_z_batch1.flatten(), ax=ax[sub_p], label=r'$x_{t+1} | x_1=-0.06$%')
sns.lineplot(x=z_batch.flatten(), y=f_z_batch2.flatten(), ax=ax[sub_p], label=r'$x_{t+1} | x_2=0.41$%')
# plot the real CDF
y = norm.pdf(z_batch,
             loc=0,
             scale=torch.std(torch.concatenate([y_train, y_val])).item())
sns.lineplot(x=z_batch.flatten(), y=y.flatten(), ax=ax[sub_p], label=r'$\mathcal{N}(0, std(x))$')
ax[sub_p].set_title('Probability Density Function')
ax[sub_p].legend(bbox_to_anchor=(1.0, 1.0))

plt.suptitle(r'(Un)Conditional NeuralCDF - $x_{t+1} | x_t$')
plt.tight_layout()
plt.savefig('../figures/neural_cdf_cond_x_t_t1_example.png', dpi=150)
plt.show()






#%% Sample points to evaluate the q-Gaussian distribution

# Fit the q-Gaussian distribution on the real data
fitted_values = qGaussian.fit(y_train.cpu().detach().numpy(), n_it=200)
fitted_q_y, fitted_mu_y, fitted_sigma_y =\
    fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
print(f"Fitted q: {fitted_q}, Fitted mu: {fitted_mu}, Fitted sigma: {fitted_sigma}")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.histplot(y_train.cpu().detach().numpy(), ax=ax, stat='probability', label='Data')

num_bins = len(ax.patches) # Number of bins
temp = ax.patches
hist_values = np.histogram(y_train.cpu().detach().numpy(), bins=num_bins, density=True)[0] #Obtain the histogram values
hist_values /= np.sum(hist_values) # Normalize the values
pdf = qGaussian.pdf(np.linspace(np.min(y_train.cpu().detach().numpy()), np.max(y_train.cpu().detach().numpy()), 10000), fitted_q, mu=fitted_mu, sigma=fitted_sigma)

sns.lineplot(x=np.linspace(np.min(y_train.cpu().detach().numpy()), np.max(y_train.cpu().detach().numpy()), 10000),
             y=pdf*np.max(hist_values)/np.max(pdf),
            ax=ax, color=sns.color_palette()[1], label='Scaled qGaussian PDF')

ax.legend()
plt.show()

# Fitted q: 1.5810021301905273, Fitted mu: -0.0005259437239173324, Fitted sigma: 0.048677821154237945



uncond_gen = mdl_un.draw(mdl_un.mdl, device, 10_000,
                         
                         z_max=np.max(y_train.cpu().detach().numpy()),
                         seed=random_seed)

cond_gen = mdl.draw(mdl.mdl, device, x_train[0].item(), 10_000,
                  z_min=torch.min(y_train).item(),
                  z_max=torch.max(y_train).item(),
                  seed=random_seed).cpu().detach().numpy()


def draw(mdl, device, n_points, z_min=-2, z_max=2, N_grid=20_000, seed=None):
    torch.manual_seed(seed) #Set the seed
    torch.cuda.manual_seed_all(seed)
    uniform_samples = torch.rand(n_points, device=device) #Sample from a uniform distribution
    
    # Create a grid of z values
    z_values = torch.linspace(z_min, z_max, N_grid, device=device, dtype=torch.float64)
    z_values = z_values.unsqueeze(1)  # Make it (num_points, 1)
    
    # Evaluate the CDF for each z in the grid
    cdf_values = mdl(z_values).squeeze(1)  # Make it (num_points,)
    
    # Function to find the z corresponding to each uniform sample using binary search
    def find_z(cdf_value):
        low, high = 0, N_grid - 1
        while low < high:
            mid = (low + high) // 2
            if cdf_values[mid] < cdf_value:
                low = mid + 1
            else:
                high = mid
        return z_values[low].item()
    
    # Apply the binary search to each uniform sample
    sampled_points = torch.tensor([find_z(u) for u in uniform_samples], device=device)
    
    return sampled_points


uncond_gen = draw(mdl_un.mdl, device, 10_000,
                  z_min=torch.min(y_train).item(),
                  z_max=torch.max(y_train).item(),
                  seed=random_seed).cpu().detach().numpy()
uncond_gen

# Plot the trajectory
y_temp = uncond_gen/100
y_temp = np.exp( y_temp.cumsum() ) * df.price[0]
plt.plot(y_temp)

# Comparison between stats
print('{:<18}{:<2}{:<15}{:<15}{:<15}'.format('Statistic', '|', 'True', 'Uncond', 'Cond'))
print('{:<65}'.format('-'*60))
print('{:<18}{:<2}{:<15}{:<15}{:<15}'.format('Min', '|', format(torch.min(y_train).item(), '.20f')[:10], format(np.min(uncond_gen), '.20f')[:10], format(np.min(cond_gen), '.20f')[:10]))
print('{:<18}{:<2}{:<15}{:<15}{:<15}'.format('Quantile 0.25', '|', format(torch.quantile(y_train, 0.25).item(), '.20f')[:10], format(np.quantile(uncond_gen, 0.25), '.20f')[:10], format(np.quantile(cond_gen, 0.25), '.20f')[:10]))
print('{:<18}{:<2}{:<15}{:<15}{:<15}'.format('Mean', '|', format(torch.mean(y_train).item(), '.20f')[:10], format(np.mean(uncond_gen), '.20f')[:10], format(np.mean(cond_gen), '.20f')[:10]))
print('{:<18}{:<2}{:<15}{:<15}{:<15}'.format('Median', '|', format(torch.median(y_train).item(), '.20f')[:10], format(np.median(uncond_gen), '.20f')[:10], format(np.median(cond_gen), '.20f')[:10]))
print('{:<18}{:<2}{:<15}{:<15}{:<15}'.format('Quantile 0.75', '|', format(torch.quantile(y_train, 0.75).item(), '.20f')[:10], format(np.quantile(uncond_gen, 0.75), '.20f')[:10], format(np.quantile(cond_gen, 0.75), '.20f')[:10]))
print('{:<18}{:<2}{:<15}{:<15}{:<15}'.format('Max', '|', format(torch.max(y_train).item(), '.20f')[:10], format(np.max(uncond_gen), '.20f')[:10], format(np.max(cond_gen), '.20f')[:10]))

# Fit the q-Gaussian distribution
import qGaussian
fitted_values = qGaussian.fit(uncond_gen, n_it=200)
fitted_q, fitted_mu, fitted_sigma =\
    fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
print(f"Fitted q: {fitted_q}, Fitted mu: {fitted_mu}, Fitted sigma: {fitted_sigma}")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.histplot(uncond_gen, ax=ax, stat='probability', label='Data')

num_bins = len(ax.patches) # Number of bins
temp = ax.patches
hist_values = np.histogram(uncond_gen, bins=num_bins, density=True)[0] #Obtain the histogram values
hist_values /= np.sum(hist_values) # Normalize the values
pdf = qGaussian.pdf(np.linspace(np.min(uncond_gen), np.max(uncond_gen), 10000), fitted_q, mu=fitted_mu, sigma=fitted_sigma)

sns.lineplot(x=np.linspace(np.min(uncond_gen), np.max(uncond_gen), 10000),
             y=pdf*np.max(hist_values)/np.max(pdf),
            ax=ax, color=sns.color_palette()[1], label='Scaled qGaussian PDF')

ax.legend()
plt.show()

q_uncond = list()
mu_uncond = list()
sigma_uncond = list()
min_uncond = list()
q25_uncond = list()
mean_uncond = list()
median_uncond = list()
q75_uncond = list()
max_uncond = list()

for temp_seed in range(51):
    uncond_gen = draw(mdl_un.mdl, device, len(y_train),
                      z_min=torch.min(y_train).item(),
                      z_max=torch.max(y_train).item(),
                      seed=temp_seed).cpu().detach().numpy()
    
    min_uncond.append(np.min(uncond_gen))
    q25_uncond.append(np.quantile(uncond_gen, 0.25))
    mean_uncond.append(np.mean(uncond_gen))
    median_uncond.append(np.median(uncond_gen))
    q75_uncond.append(np.quantile(uncond_gen, 0.75))
    max_uncond.append(np.max(uncond_gen))

    fitted_values = qGaussian.fit(uncond_gen, n_it=200)
    fitted_q, fitted_mu, fitted_sigma =\
        fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
    q_uncond.append( fitted_q )
    mu_uncond.append( fitted_mu )
    sigma_uncond.append( fitted_sigma )
    print(f"Fitted q: {fitted_q}, Fitted mu: {fitted_mu}, Fitted sigma: {fitted_sigma}")













#Apply the Kolmogorov-Smirnov test - one sample
from scipy.stats import kstest

def mdl4test(mdl, x):
    if len(x.shape) == 1:
        return mdl(torch.tensor(x.reshape(-1,1), dtype=torch.float64).to(device)).cpu().detach().numpy().flatten()
    else:
        return mdl(torch.tensor(x, dtype=torch.float64).to(device)).cpu().detach().numpy()

kstest(y_train.cpu().detach().numpy(),
       lambda x: mdl4test(mdl_un.mdl, x)
       )

# Fitted q: 1.3321268174571121, Fitted mu: 0.00013864594355770385, Fitted sigma: 0.06335465703970712



def sample_conditional_path(mdl, device, x0, num_points, z_min=-10.0,
                            z_max=10.0, N_grid=20_000, seed=None):
    """
    Generate a path of points from a conditional distribution represented by a neural network mdl.
    
    Args:
        mdl (nn.Module): Neural network representing the conditional CDF.
        x0 (float): Starting point x_0.
        num_points (int): Number of points to generate in the path.
        z_min (float): Minimum value of z to consider in the search.
        z_max (float): Maximum value of z to consider in the search.
        N_grid (int): Number of points to use in the search grid.
        seed (int): Seed for the random number generator.
        
    Returns:
        torch.Tensor: Tensor of generated points.
    """
    if not isinstance(seed, type(None)):
        torch.manual_seed(seed) #Set the seed
        torch.cuda.manual_seed_all(seed)
    
    # Create a grid of z values
    z_values = torch.linspace(z_min, z_max, N_grid, device=device, dtype=torch.float64)
    z_values = z_values.unsqueeze(1)  # Make it (N_grid, 1)
    
    # Initialize the path with the starting point
    path = [x0]
    
    for _ in range(1, num_points):
        # Get the previous point
        x_prev = path[-1]
        
        # Generate a uniform random sample in [0, 1]
        uniform_sample = torch.rand(1, device=device).item()
        
        # Create the input tensor for the model (z_values, x_prev)
        inputs = torch.cat([z_values, torch.full((N_grid, 1), x_prev, device=device)], dim=1)
        
        # Evaluate the CDF for each z in the grid given x_prev
        with torch.no_grad():
            cdf_values = mdl(inputs).squeeze(1)  # Make it (N_grid,)
        
        # Function to find the z corresponding to the uniform sample using binary search
        def find_z(cdf_value):
            low, high = 0, N_grid - 1
            while low < high:
                mid = (low + high) // 2
                if cdf_values[mid] < cdf_value:
                    low = mid + 1
                else:
                    high = mid
            return z_values[low].item()
        
        # Find the z corresponding to the uniform sample
        next_point = find_z(uniform_sample)
        
        # Append the generated point to the path
        path.append(next_point)
    
    return torch.tensor(path, device=device)

cond_gen = sample_conditional_path(mdl.mdl, device, x_train[0].item(), 10_000,
                  z_min=torch.min(y_train).item(),
                  z_max=torch.max(y_train).item(),
                  seed=random_seed).cpu().detach().numpy()
cond_gen

# Plot the trajectory
y_temp = cond_gen/100
y_temp = np.exp( y_temp.cumsum() ) * df.price[0]
plt.plot(y_temp)

# Comparison between stats
print('{:<15}{:<15}{:<15}'.format('Statistic', 'True', 'Generated'))
print('{:<15}{:<15}{:<15}'.format('Min', round(torch.min(y_train).item(), 5), round(np.min(cond_gen), 5)))
print('{:<15}{:<15}{:<15}'.format('Quantile 0.25', round(torch.quantile(y_train, 0.25).item(), 5), round(np.quantile(cond_gen, 0.25), 5)))
print('{:<15}{:<15}{:<15}'.format('Mean', round(torch.mean(y_train).item(), 5), round(np.mean(cond_gen), 5)))
print('{:<15}{:<15}{:<15}'.format('Median', round(torch.median(y_train).item(), 5), round(np.median(cond_gen), 5)))
print('{:<15}{:<15}{:<15}'.format('Quantile 0.75', round(torch.quantile(y_train, 0.75).item(), 5), round(np.quantile(cond_gen, 0.75), 5)))
print('{:<15}{:<15}{:<15}'.format('Max', round(torch.max(y_train).item(), 5), round(np.max(cond_gen), 5)))

# Fit the q-Gaussian distribution
import qGaussian
fitted_values = qGaussian.fit(cond_gen, n_it=200)
fitted_q, fitted_mu, fitted_sigma =\
    fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
print(f"Fitted q: {fitted_q}, Fitted mu: {fitted_mu}, Fitted sigma: {fitted_sigma}")

#Fitted q: 1.7057712988429214, Fitted mu: 0.005123073221874671, Fitted sigma: 0.04211340714683504

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.histplot(cond_gen, ax=ax, stat='probability', label='Data')

num_bins = len(ax.patches) # Number of bins
temp = ax.patches
hist_values = np.histogram(cond_gen, bins=num_bins, density=True)[0] #Obtain the histogram values
hist_values /= np.sum(hist_values) # Normalize the values
pdf = qGaussian.pdf(np.linspace(np.min(cond_gen), np.max(cond_gen), 10000), fitted_q, mu=fitted_mu, sigma=fitted_sigma)

sns.lineplot(x=np.linspace(np.min(cond_gen), np.max(cond_gen), 10000),
             y=pdf*np.max(hist_values)/np.max(pdf),
            ax=ax, color=sns.color_palette()[1], label='Scaled qGaussian PDF')

ax.legend()
plt.show()

q_cond = list()
mu_cond = list()
sigma_cond = list()
min_cond = list()
q25_cond = list()
mean_cond = list()
median_cond = list()
q75_cond = list()
max_cond = list()

for temp_seed in range(51):
    cond_gen = sample_conditional_path(mdl.mdl, device, x_train[0].item(), len(y_train),
                    z_min=torch.min(y_train).item(),
                    z_max=torch.max(y_train).item(),
                    seed=temp_seed).cpu().detach().numpy()
    
    min_cond.append(np.min(cond_gen))
    q25_cond.append(np.quantile(cond_gen, 0.25))
    mean_cond.append(np.mean(cond_gen))
    median_cond.append(np.median(cond_gen))
    q75_cond.append(np.quantile(cond_gen, 0.75))
    max_cond.append(np.max(cond_gen))

    fitted_values = qGaussian.fit(cond_gen, n_it=200)
    fitted_q, fitted_mu, fitted_sigma =\
        fitted_values['q'], fitted_values['mu'], fitted_values['sigma']
    q_cond.append( fitted_q )
    mu_cond.append( fitted_mu )
    sigma_cond.append( fitted_sigma )
    print(f"Fitted q: {fitted_q}, Fitted mu: {fitted_mu}, Fitted sigma: {fitted_sigma}")



base = '{:<15}{:<2}{:<15}{:<2}{:<15}{:<15}{:<2}{:<15}{:<15}'
print('{:<15}{:<2}{:<15}{:<2}{:^30}{:<2}{:^20}'.format('Statistic', '|', 'True', '|', 'Unconditional', '|', 'Conditional'))
print(base.format('', '|', '', '|', 'mean', 'std', '|', 'mean', 'std'))
print('{:<94}'.format('-'*90))
for stat_name, temp_real, temp_uncond, temp_cond in zip(
    ['Min', 'Quantile 0.25', 'Mean', 'Median', 'Quantile 0.75',
     'Max', 'Fitted q', 'Fitted mu', 'Fitted sigma'],
    [torch.min(y_train).item(), torch.quantile(y_train, 0.25),
     torch.mean(y_train).item(), torch.median(y_train).item(),
     torch.quantile(y_train, 0.75), torch.max(y_train).item(),
     fitted_q_y, fitted_mu_y, fitted_sigma_y],
    [min_uncond, q25_uncond, mean_uncond, median_uncond,
     q75_uncond, max_uncond, q_uncond, mu_uncond, sigma_uncond],
     [min_cond, q25_cond, mean_cond, median_cond,
      q75_cond, max_cond, q_cond, mu_cond, sigma_cond]):
    print(base.format(stat_name, '|', format(temp_real, '.20f')[:10], '|',
                    format(np.mean(temp_uncond), '.20f')[:10],
                    format(np.std(temp_uncond), '.20f')[:10], '|',
                    format(np.mean(temp_cond), '.20f')[:10],
                    format(np.std(temp_cond), '.20f')[:10]))
    

base = '{:<15}{:<2}{:<15}{:<2}{:<15}{:<15}{:<2}{:<15}{:<15}'
print('{:<15}{:<2}{:<15}{:<2}{:^30}{:<2}{:^20}'.format('Statistic', '|', 'True', '|', 'Unconditional', '|', 'Conditional'))
print(base.format('', '|', '', '|', 'mean', 'std', '|', 'mean', 'std'))
print('{:<94}'.format('-'*90))
for stat_name, temp_real, temp_uncond, temp_cond in zip(
    ['Min', 'Quantile 0.25', 'Mean', 'Median', 'Quantile 0.75', 'Max'],
    [torch.min(y_train).item(), torch.quantile(y_train, 0.25),
     torch.mean(y_train).item(), torch.median(y_train).item(),
     torch.quantile(y_train, 0.75), torch.max(y_train).item()],
    [min_uncond, q25_uncond, mean_uncond, median_uncond, q75_uncond, max_uncond],
     [min_cond, q25_cond, mean_cond, median_cond, q75_cond, max_cond]):
    print(base.format(stat_name, '|', format(temp_real, '.20f')[:10], '|',
                    format(np.mean(temp_uncond), '.20f')[:10],
                    format(np.std(temp_uncond), '.20f')[:10], '|',
                    format(np.mean(temp_cond), '.20f')[:10],
                    format(np.std(temp_cond), '.20f')[:10]))


base = '{:<15}{:<2}{:<15}{:<2}{:<15}{:<15}{:<2}{:<15}{:<15}'
print('{:<15}{:<2}{:<15}{:<2}{:^30}{:<2}{:^20}'.format('Statistic', '|', 'True', '|', 'Unconditional', '|', 'Conditional'))
print(base.format('', '|', '', '|', 'mean', 'std', '|', 'mean', 'std'))
print('{:<94}'.format('-'*90))
for stat_name, temp_real, temp_uncond, temp_cond in zip(
    ['Fitted q', 'Fitted mu', 'Fitted sigma'],
    [fitted_q_y, fitted_mu_y, fitted_sigma_y],
    [q_uncond, mu_uncond, sigma_uncond],
     [q_cond, mu_cond, sigma_cond]):
    print(base.format(stat_name, '|', format(temp_real, '.20f')[:10], '|',
                    format(np.mean(temp_uncond), '.20f')[:10],
                    format(np.std(temp_uncond), '.20f')[:10], '|',
                    format(np.mean(temp_cond), '.20f')[:10],
                    format(np.std(temp_cond), '.20f')[:10]))
















#Apply the Kolmogorov-Smirnov test - one sample
from scipy.stats import kstest

def mdl4test(mdl, x):
    if len(x.shape) == 1:
        return mdl(torch.tensor(x.reshape(-1,1), dtype=torch.float64).to(device)).cpu().detach().numpy().flatten()
    else:
        return mdl(torch.tensor(x, dtype=torch.float64).to(device)).cpu().detach().numpy()

kstest(y_train.cpu().detach().numpy(),
       lambda x: mdl4test(mdl_un.mdl, x)
       )


uncond_gen

kstest(uncond_gen,
       lambda x: mdl4test(mdl_un.mdl, x)
       )



def mdl4test(mdl, x0, z):
    mdl(torch.tensor([x0].reshape(-1,1), dtype=torch.float64).to(device)).cpu().detach().numpy().flatten()
    if len(x.shape) == 1:
        return mdl(torch.tensor(x.reshape(-1,1), dtype=torch.float64).to(device)).cpu().detach().numpy().flatten()
    else:
        return mdl(torch.tensor(x, dtype=torch.float64).to(device)).cpu().detach().numpy()

kstest(y_train.cpu().detach().numpy(),
       lambda z: mdl4test(mdl.mdl, z)
       )












def draw(mdl, device, n_points, seed=None):
    from scipy.optimize import fsolve

    np.random.seed(seed) #Set the seed
    x = np.random.uniform(0, 1, n_points) #Sample from a uniform distribution
    initial_guess = np.zeros_like(x)

    # Use fsolve to find the root of inverse_function
    return fsolve(
        lambda y: mdl(torch.tensor(
            y.reshape(-1,1),
            dtype=torch.float64).to(device)).cpu().detach().numpy().flatten() - x,
        initial_guess)





