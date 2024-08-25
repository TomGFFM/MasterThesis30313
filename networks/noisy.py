import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    A fully connected layer that adds learnable noise to the weights and biases for exploration in reinforcement learning.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        sigma_init (float, optional): Initial value for the standard deviation of the noise. Default is 0.017.

    Attributes:
        weight_mu (torch.nn.Parameter): The mean (mu) of the weights.
        weight_sigma (torch.nn.Parameter): The standard deviation (sigma) of the weights.
        weight_epsilon (torch.Tensor): The noise applied to the weights.
        bias_mu (torch.nn.Parameter): The mean (mu) of the biases.
        bias_sigma (torch.nn.Parameter): The standard deviation (sigma) of the biases.
        bias_epsilon (torch.Tensor): The noise applied to the biases.
    """

    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        # Store the number of input and output features
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Initialize the mean (mu) of the weights with zeros
        self.weight_mu = nn.Parameter(torch.full((out_features, in_features), 0.0))
        # Initialize the standard deviation (sigma) of the weights
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # Register a buffer to hold the noise (epsilon) applied to the weights
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        # Initialize the mean (mu) of the biases with zeros
        self.bias_mu = nn.Parameter(torch.full((out_features,), 0.0))
        # Initialize the standard deviation (sigma) of the biases
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init))
        # Register a buffer to hold the noise (epsilon) applied to the biases
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        # Initialize the parameters using a method that takes into account the input size
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters using a uniform distribution."""
        # Calculate a range for uniform initialization based on input size
        mu_range = 1 / self.in_features**0.5
        # Initialize weights and biases within the calculated range
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

    def forward(self, x):
        """Forward pass with noisy weights and biases."""
        # Sample new noise values for weights and biases
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

        # Compute the noisy weights and biases by adding the noise to the mean
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        # Apply the linear transformation with the noisy weights and biases
        return F.linear(x, weight, bias)
