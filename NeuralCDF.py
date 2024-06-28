import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import dask.dataframe as dd
from data_processing import plot_price_and_liquidity, slice_price_by_liquidity_dists
from input_data_creation_cython import optimize_dataframe

class NeuralCDF(nn.Module):
    def __init__(self, n1, n2, lstm_hidden_dim, lstm_layers, dense_layers, output_dim, dropout=0.5):
        """
        Initialize the CustomNN model.

        Args:
            n1 (int): Dimension of vector p.
            n2 (int): Dimension of vector l.
            lstm_hidden_dim (int): Number of features in the hidden state of LSTM.
            lstm_layers (int): Number of LSTM layers.
            dense_layers (list): List of integers where each integer represents the number of neurons in that dense layer.
            output_dim (int): Dimension of the output layer.
            dropout (float): Dropout probability to be applied after each dense layer.
        """
        super(CustomNN, self).__init__()
        # batch_first=True indicates that the input data will have the shape (batch_size, sequence_length, n1)
        # while the output data will have the shape (batch_size, sequence_length, lstm_hidden_dim)
        self.lstm = nn.LSTM(input_size=n1, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # Create a list of dense layers using nn.ModuleList
        self.dense_layers = nn.ModuleList()
        
        # First dense layer (input: lstm_hidden_dim + n2)
        self.dense_layers.append(nn.Linear(lstm_hidden_dim + n2, dense_layers[0]))
        
        # Hidden dense layers
        for i in range(1, len(dense_layers)):
            self.dense_layers.append(nn.Linear(dense_layers[i-1], dense_layers[i]))
        
        # Output layer
        self.output_layer = nn.Linear(dense_layers[-1], output_dim)
        
        # Initialize weights using Glorot initialization
        for layer in self.dense_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, p, l):
        lstm_out, _ = self.lstm(p)  # Assuming p has shape (batch_size, sequence_length, n1)
        lstm_out = lstm_out[:, -1, :]  # Taking the last hidden state
        x = torch.cat((lstm_out, l), dim=1)  # Concatenating lstm output with vector l
        
        for layer in self.dense_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        x = torch.sigmoid(self.output_layer(x))
        return x
    
    def crps_loss(self, z_batch, y_batch, l=3, num_points=200):
        '''
        Compute the CRPS loss for the model and input batches.
        
        Parameters:
        z_batch: torch.Tensor - Batch of z values (batch_size, dim)
        y_batch: torch.Tensor - Batch of y values (batch_size, dim)
        l: int - Limits for integration (default=3)
        num_points: int - Number of points for integration (default=200)
        
        Returns:
        torch.Tensor - The CRPS loss
        '''
        # Pass z_batch through the neural network
        F_z_batch = self.forward(z_batch)
        
        # Integral argument computation
        f = torch.where(z_batch >= y_batch, (F_z_batch - 1) ** 2, F_z_batch ** 2)
        
        # dx for integration
        dx = 2 * l / (num_points - 1)
        
        # Integral approximation
        integral = (torch.sum(f) + 0.5 * (torch.sum(f[:, 0]) + torch.sum(f[:, -1]))) * dx
        
        return integral
    
    def monotonic_loss(self, z_batch, F_z_batch):
        '''Implement the monotonic loss for the NN.'''
        f_z_batch = torch.autograd.grad(F_z_batch, z_batch, grad_outputs=torch.ones_like(F_z_batch), create_graph=True, retain_graph=True)[0]
        return torch.where(f_z_batch > 0, torch.zeros_like(f_z_batch), f_z_batch).sum()

    def l2_regularization(self):
        '''Implement the L2 regularization for the NN.'''
        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.sum(param ** 2)
        return l2_reg

if __name__ == '__main__':

    # # Create a graph plot of the model
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # # Render the graph to a file
    # dot.render("images/custom_nn_model", format="png")
    pass