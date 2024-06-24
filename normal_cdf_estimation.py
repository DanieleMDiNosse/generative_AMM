import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import os
from utils import print_
from datetime import datetime
from scipy.stats import norm
from torch.utils.tensorboard import SummaryWriter
import argparse

class reverse_autoencoder(nn.Module):
    def __init__(self, input_dim=1, n_layers=3, dropout=0.2):
        super(reverse_autoencoder, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        
        hidden_up = [2**(i+3) for i in range(1, n_layers+1)]
        hidden_down = hidden_up[::-1]
        hidden_down.append(input_dim)
        
        self.layers = nn.ModuleList()
        for i in range(2 * n_layers):
            if i < n_layers:
                self.layers.append(nn.Linear(input_dim, hidden_up[i]))
                input_dim = hidden_up[i]
            else:
                input_dim = hidden_down[i - n_layers]
                self.layers.append(nn.Linear(input_dim, hidden_down[i + 1 - n_layers]))
                
            self.layers.append(nn.Dropout(self.dropout))
        
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for i in range(2 * self.n_layers):
            x = self.layers[i * 2](x)
            x = torch.relu(x)
            x = self.layers[i * 2 + 1](x)
        
        x = self.layers[-1](x)
        x = self.output_activation(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self, input_dim=1, n_layers=2, dropout=0.2):
        super(SimpleNet, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        hidden_neurons = [2**(i+3) for i in range(1, n_layers+1)][::-1]
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_neurons[i]))
            else:
                self.layers.append(nn.Linear(hidden_neurons[i-1], hidden_neurons[i]))
            
            self.layers.append(nn.Dropout(self.dropout))
                
        self.layers.append(nn.Linear(hidden_neurons[-1], input_dim))
        self.output_activation = nn.Sigmoid()

        # Apply Glorot initialization to each layer
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i * 2](x)
            x = torch.relu(x)
            x = self.layers[i * 2 + 1](x)
        x = self.layers[-1](x)
        x = self.output_activation(x)
        return x

def integral_argument(z_batch, F_z_batch, y_batch):
    '''Integral argument for the loss function.'''
    f = torch.where(z_batch >= y_batch, (F_z_batch - 1) ** 2, F_z_batch ** 2)
    return f

def scoring_loss(z_batch, F_z_batch, y_batch, l=3, num_points=200):
    '''Implement the scoring loss for the NN.'''
    f = integral_argument(z_batch, F_z_batch, y_batch)
    dx = 2 * l / (num_points - 1)
    integral = (torch.sum(f) + 0.5 * (f[0] + f[-1])) * dx
    return integral

def monotonic_loss(z_batch, F_z_batch):
    '''Implement the monotonic loss for the NN.'''
    f_z_batch = torch.autograd.grad(F_z_batch, z_batch, grad_outputs=torch.ones_like(F_z_batch), create_graph=True, retain_graph=True)[0]
    return torch.where(f_z_batch > 0, torch.zeros_like(f_z_batch), f_z_batch).sum()

def monotonic_loss_soft(z_batch, F_z_batch):
    '''Implement the monotonic loss for the NN with a soft constraint.'''
    f_z_batch = torch.autograd.grad(F_z_batch, z_batch, grad_outputs=torch.ones_like(F_z_batch), create_graph=True, retain_graph=True)[0]
    penalty = torch.relu(f_z_batch)  # Use ReLU to penalize positive gradients
    return (penalty).sum()

def l2_regularization(net):
    '''Implement the L2 regularization for the NN.'''
    l2_reg = torch.tensor(0.)
    for param in net.parameters():
        l2_reg += torch.sum(param ** 2)
    return l2_reg

def sample_data(batch_size, input_dim=1, l=3, requires_grad=False):
    '''Sample uniform data.'''
    uniform_batch = 2 * l * torch.rand(batch_size, input_dim) - l
    return uniform_batch.clone().detach().requires_grad_(requires_grad)

def sample_normal(batch_size, input_dim=1):
    '''Sample real data from a normal distribution.'''
    return torch.randn(batch_size, input_dim)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num_epochs', type=int, default=10000, help='Number of epochs')
    parser.add_argument('-p', '--max_patience', type=int, default=1000, help='Number of epochs to wait for early stopping')
    parser.add_argument('-s', '--n_scoring_points', type=int, default=1000, help='Number of points to use in the scoring loss')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('-n', '--network', type=str, default='simple', help='Network to use. Possible values: simple, reverse_autoencoder')
    parser.add_argument('-l', '--n_layers', type=int, default=2, help='Number of layers in the network')
    parser.add_argument('-o', '--optimizer', type=str, default='adam', help='Optimizer to use. Possible values: adam, sgd, rmsprop')
    parser.add_argument('-d', '--dropout', type=float, default=0.2, help='Dropout rate')
    args = parser.parse_args()

    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('models'):
        os.makedirs('models')

    print_(f"Date:L\n\t{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    print_(f'Args:\n\t{args}')

    # Tensorboard writer
    os.makedirs(f'runs/normal_cdf_estimation/{os.getpid()}_{args}')
    writer = SummaryWriter(f'runs/normal_cdf_estimation/{os.getpid()}')
    
    # Initialize the network with input dimensionality
    input_dim = 1
    if args.network == 'simple':
        net = SimpleNet(input_dim=input_dim, n_layers=args.n_layers, dropout=args.dropout)
    elif args.network == 'reverse_autoencoder':
        net = reverse_autoencoder(input_dim=input_dim, n_layers=args.n_layers, dropout=args.dropout)
    # print_ the network
    print_(f'Network:\n\t{net}')

    # Hyperparameters
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    n_scoring_points = args.n_scoring_points
    lr = args.learning_rate
    l = 4.5

    # Adam optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=lr)

    # Early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    max_patience = args.max_patience
    patience_count = 0
    val_loss = torch.tensor(float('inf'))
    
    # Training loop
    for epoch in range(num_epochs):
        net.train() # Put network in 'training mode'
        optimizer.zero_grad() # Zero out the gradients. Do this every iteration to avoid accumulating gradients (start from scratch every iteration)

        # Sample data
        z_batch = sample_data(batch_size, input_dim, l=l, requires_grad=True) # (batch_size, input_dim)
        y_batch = sample_normal(batch_size, input_dim) # (batch_size, input_dim)
        # Forward pass
        F_z_batch = net(z_batch) # (batch_size, output_dim=input_dim)

        # Compute the loss
        score_loss = scoring_loss(z_batch, F_z_batch, y_batch, l=l, num_points=n_scoring_points)
        monotonic_penalty = monotonic_loss_soft(z_batch, F_z_batch)
        l2_reg = l2_regularization(net)
        # Compute the total loss
        loss = score_loss + 0.01 * monotonic_penalty + 0.01 * l2_reg
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print_(f"Epoch {epoch}, Loss: {loss.item()}, Val loss: {val_loss.item()}")
            writer.add_scalar('Loss/train', loss.item(), epoch)
            # log in tensorboard the weights and biases
            for name, param in net.named_parameters():
                writer.add_histogram(name, param, epoch)
                writer.add_histogram(f'{name}.grad', param.grad, epoch)

        # Validation
        net.eval() # Put network in 'evaluation mode'. Gradient computation is turned off
        # sample data for the validation set
        z_batch_val = sample_data(batch_size=batch_size, input_dim=1, requires_grad=True)
        y_batch_val = sample_normal(batch_size=batch_size, input_dim=1)
        F_z_batch_val = net(z_batch_val) # (batch_size, input_dim)
        score_val_loss = scoring_loss(z_batch_val, F_z_batch_val, y_batch_val, l=l, num_points=n_scoring_points)
        monotonic_penalty_val = monotonic_loss(z_batch_val, F_z_batch_val)
        l2_reg_val = l2_regularization(net)
        val_loss = score_val_loss + 0.01 * monotonic_penalty_val + 0.01 * l2_reg_val
        # log the validation loss in tensorboard
        writer.add_scalar('Loss/val', val_loss.item(), epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
        if patience_count > max_patience:
            # Save the best model
            torch.save(net.state_dict(), f'models/best_model_{args.network}_{os.getpid()}.pt')
            print_(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}. Best val loss: {best_val_loss}")
            break
        if epoch == num_epochs - 1:
            torch.save(net.state_dict(), f'models/best_model_{args.network}_{os.getpid()}.pt')
            print_(f"Training completed at epoch {epoch}. Best epoch: {best_epoch}. Best val loss: {best_val_loss}")

    
    # Test the network on a test set
    # Load the best model weights
    net.load_state_dict(torch.load(f'models/best_model_{args.network}_{os.getpid()}.pt'))
    net.eval()
    z_batch_test = sample_data(batch_size=batch_size, input_dim=1, requires_grad=True)
    y_batch_test = sample_normal(batch_size=batch_size, input_dim=1)
    F_z_batch_test = net(z_batch_test)
    f_z_batch_test = torch.autograd.grad(F_z_batch_test, z_batch_test, grad_outputs=torch.ones_like(F_z_batch_test))[0]
    # sort F_z_batch
    F_z_batch_test, _ = torch.sort(F_z_batch_test)
    f_z_batch_test, _ = torch.sort(f_z_batch_test)
    # convert to numpy both z_batch and F_z_batch
    F_z_batch_test = F_z_batch_test.detach().numpy()
    f_z_batch_test = f_z_batch_test.detach().numpy()
    z_batch_test = z_batch_test.detach().numpy()

    # plot the CDF
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(z_batch_test, F_z_batch_test, color='red', alpha=0.7, label='Estimated CDF', s=5)
    y = norm.cdf(z_batch_test)
    axes[0].scatter(z_batch_test, y, color='green', alpha=0.7, label='Real CDF', s=5)
    axes[0].legend()
    # plot the PDF
    axes[1].scatter(z_batch_test, f_z_batch_test, color='red', alpha=0.7, label='Estimated PDF', s=5)
    y = norm.pdf(z_batch_test)
    axes[1].scatter(z_batch_test, y, color='green', alpha=0.7, label='Real PDF', s=5)
    axes[1].legend()
    if not os.path.exists('images'):
        os.makedirs('images')
    fig.savefig(f'images/{os.getpid()}_CDF_normal_{args}.png')
    
    print(os.getpid())



