import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.tensorboard import SummaryWriter
import argparse

class SimpleNet(nn.Module):
    def __init__(self, input_dim=1):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer
        self.fc3 = nn.Linear(64, 32)         # Third hidden layer`
        self.fc4 = nn.Linear(32, 16)         # Fourth hidden layer
        self.fc5 = nn.Linear(16, input_dim)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def integral_argument(z_batch, F_z_batch, y_batch, l=3):
    '''Integral argument for the loss function.'''
    f = []
    for z, F_z, y in zip(z_batch, F_z_batch, y_batch):
        if z >= y:
            f.append((F_z - 1)**2)
        else:
            f.append(F_z**2)
    return torch.stack(f)

def scoring_loss(z_batch, F_z_batch, y_batch, l=3, num_points=200):
    '''Implement the scoring loss for the NN.'''
    x = torch.linspace(-l, l, num_points)
    eval = integral_argument(z_batch, F_z_batch, y_batch, l)
    dx = 2 * l / (num_points - 1)
    integral = (torch.sum(eval) - 0.5 * (eval[0] + eval[-1])) * dx
    return integral

def sample_data(batch_size, input_dim=1, l=3, requires_grad=False):
    # Uniform distribution in [-l, l]
    uniform_batch = 2 * l * torch.rand(batch_size, input_dim) - l
    return torch.tensor(uniform_batch, requires_grad=requires_grad)

def sample_real_data(batch_size, input_dim=1):
    real_batch = torch.randn(batch_size, input_dim)
    return real_batch

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    # Number of epochs and batch size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    writer = SummaryWriter('runs/normal_cdf_estimation')
    # Initialize the network with input dimensionality
    input_dim = 1  # Adjust based on your input feature size
    net = SimpleNet(input_dim=input_dim)
    # Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    l = 4

    # Training loop
    for epoch in range(num_epochs):
        net.train()
        optimizer.zero_grad()

        # Sample data
        z_batch = sample_data(batch_size, input_dim, l=l) # (batch_size, input_dim)
        y_batch = sample_real_data(batch_size, input_dim) # (batch_size, input_dim)
        F_z_batch = net(z_batch) # (batch_size, output_dim=input_dim)

        # Compute the loss
        loss = scoring_loss(z_batch, F_z_batch, y_batch, l=l, num_points=500) 

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            writer.add_scalar('Loss/train', loss.item(), epoch)
            # log in tensorboard the weights and biases
            for name, param in net.named_parameters():
                writer.add_histogram(name, param, epoch)
                writer.add_histogram(f'{name}.grad', param.grad, epoch)

    
    # test the network
    net.eval()
    z_batch = sample_data(batch_size=200, input_dim=1, requires_grad=True)
    F_z_batch = net(z_batch)
    f_z_batch = torch.autograd.grad(F_z_batch, z_batch, grad_outputs=torch.ones_like(F_z_batch))[0]
    # sort F_z_batch
    F_z_batch, _ = torch.sort(F_z_batch)
    f_z_batch, _ = torch.sort(f_z_batch)
    # convert to numpy both z_batch and F_z_batch
    F_z_batch = F_z_batch.detach().numpy()
    f_z_batch = f_z_batch.detach().numpy()
    z_batch = z_batch.detach().numpy()

    # plot the CDF
    plt.scatter(z_batch, F_z_batch)

    # plot the real CDF
    y = norm.cdf(z_batch)
    plt.scatter(z_batch, y, color='red')


    plt.figure()
    plt.scatter(z_batch, f_z_batch)

    # plot the real CDF
    y = norm.pdf(z_batch)
    plt.scatter(z_batch, y, color='red')
    plt.show()



