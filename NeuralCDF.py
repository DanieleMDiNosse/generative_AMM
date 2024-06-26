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

class CustomNN(nn.Module):
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
        self.lstm = nn.LSTM(input_size=n1, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
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

if __name__ == '__main__':

    # # Create a graph plot of the model
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # # Render the graph to a file
    # dot.render("images/custom_nn_model", format="png")
    pass