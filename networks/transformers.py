import logging
import numpy as np

import torch
import torch.nn as nn
from networks import CNNExtractor, NoisyLinear


class DDQAugmentedTransformerNN(nn.Module):
    """
    Integrates convolutional layers with attention-augmented convolution and transformer
    encoders to enhance feature detection capabilities and manage long-range dependencies.
    This hybrid model is designed for tasks requiring both local and global context awareness.

    Attributes:
        cnnex (CNNExtractor): Initial convolution layer to extract preliminary features from input.
        transformer_encoders (nn.ModuleList): List of transformer encoder layers to process sequence data.
        advantage (nn.Sequential): Fully connected layers for advantage value computation.
        value (nn.Sequential): Fully connected layers for state value computation.

    Args:
        input_shape (tuple): Shape of the input (channels, height, width).
        num_actions (int): Number of possible actions or outputs the model can generate.
        num_heads (int): Number of attention heads in each transformer layer.
        num_layers (int): Number of transformer encoder layers.
        conv_channels (list): The number of output channels for each convolutional layer in CNNExtractor.
                              Default: [32, 64, 128, 256]
        save_images (bool): Whether to save the output images of each convolutional layer in CNNExtractor.
                            Default: False
        output_dir (str): The directory to save the output images from CNNExtractor.
                          Default: 'output'
    """
    def __init__(self, input_shape: tuple, num_actions, num_heads=8, num_layers=6, size_linear_layers=512,
                 conv_channels: list = [32, 64, 128, 256], dropout_linear: float = 0.3, save_images: bool = False, output_dir: str = './output'):
        super(DDQAugmentedTransformerNN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self._model_name = 'DDQAugmentedTransformerNN'

        # cnn layer for feature extraction
        self.cnnex = CNNExtractor(input_shape, conv_channels, save_images, output_dir)

        # calculate the dimensionality of the feature vector
        self.num_channels = conv_channels[-1]  # number of channels from the last CNN layer
        self.seq_length = self.calculate_seq_length(input_shape)  # calculate the sequence length based on input shape

        # stack of transformer encoder layers for deep sequence processing
        self.transformer_encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.num_channels, nhead=num_heads) for _ in range(num_layers)
        ])

        # Define fully connected layers for advantage value computation
        self.advantage = nn.Sequential(
            nn.Linear(self.num_channels * self.seq_length, size_linear_layers),
            nn.ReLU(),
            nn.Dropout(dropout_linear),
            nn.Linear(size_linear_layers, self.num_actions)
        )

        # Define fully connected layers for state value computation
        self.value = nn.Sequential(
            nn.Linear(self.num_channels * self.seq_length, size_linear_layers),
            nn.ReLU(),
            nn.Dropout(dropout_linear),
            nn.Linear(size_linear_layers, 1)
        )

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, name: str):
        self._model_name = name

    def forward(self, x):
        # extract features using CNN
        x = self.cnnex(x)

        # Debugging-Ausgabe
        logging.debug(f"Shape after CNN: {x.shape}")

        # get batch size
        batch_size = x.size(0)

        # Calculate the correct seq_length based on the actual output shape of CNN
        seq_length = x.size(2) * x.size(3)
        logging.debug(f"Calculated seq_length: {seq_length}")

        # reshape the features to include sequence dimension
        x = x.view(batch_size, self.num_channels, self.seq_length)
        logging.debug(f"Shape after view: {x.shape}")
        x = x.permute(0, 2, 1)
        logging.debug(f"Shape after permute: {x.shape}")

        # process through each transformer encoder layer
        for encoder in self.transformer_encoders:
            x = encoder(x)
            logging.debug(f"Shape after transformer encoder: {x.shape}")

        # remove the sequence dimension post transformer
        x = x.view(batch_size, -1)
        logging.debug(f"Shape after view to remove sequence dimension: {x.shape}")

        # apply layer normalization
        x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
        logging.debug(f"Shape after layer normalization: {x.shape}")

        # compute advantage values
        advantage = self.advantage(x)
        logging.debug(f"Shape after advantage calculation: {advantage.shape}")

        # compute state value
        value = self.value(x)
        logging.debug(f"Shape after value calculation: {value.shape}")

        # combine advantage and value to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        logging.debug(f"Shape of Q-values: {q_values.shape}")

        return q_values.squeeze()

    def calculate_seq_length(self, input_shape: tuple) -> int:
        """
        Calculate the sequence length based on the input shape and the CNN architecture.

        Args:
            input_shape (tuple): The shape of the input images (channels, height, width).

        Returns:
            int: The calculated sequence length.
        """
        # Create a dummy input tensor
        dummy_input = torch.zeros((1, *input_shape))

        # Pass the dummy input through the CNN
        dummy_output = self.cnnex(dummy_input)
        logging.debug(f"Shape of dummy output: {dummy_output.shape}")

        # Calculate the sequence length based on the output shape
        seq_length = dummy_output.size(2) * dummy_output.size(3)
        logging.debug(f"Calculated seq_length in calculate_seq_length: {seq_length}")  # Debugging-Ausgabe

        return seq_length


class DDQAugmentedNoisyTransformerNN(nn.Module):
    """
        Integrates convolutional layers with attention-augmented convolution and transformer
        encoders to enhance feature detection capabilities and manage long-range dependencies.
        This hybrid model is designed for tasks requiring both local and global context awareness,
        incorporating Noisy Linear layers for exploration.

        Attributes:
            cnnex (CNNExtractor): Initial convolution layer to extract preliminary features from input.
            transformer_encoders (nn.ModuleList): List of transformer encoder layers to process sequence data.
            advantage (nn.Sequential): Fully connected layers with NoisyLinear for advantage value computation.
            value (nn.Sequential): Fully connected layers with NoisyLinear for state value computation.

        Args:
            input_shape (tuple): Shape of the input (channels, height, width).
            num_actions (int): Number of possible actions or outputs the model can generate.
            num_heads (int): Number of attention heads in each transformer layer.
            num_layers (int): Number of transformer encoder layers.
            conv_channels (list): The number of output channels for each convolutional layer in CNNExtractor.
                                  Default: [32, 64, 128, 256]
            save_images (bool): Whether to save the output images of each convolutional layer in CNNExtractor.
                                Default: False
            output_dir (str): The directory to save the output images from CNNExtractor.
                              Default: 'output'
    """
    def __init__(self,
                 input_shape: tuple,
                 num_actions,
                 num_heads=8,
                 num_layers=6,
                 size_linear_layers=512,
                 conv_channels: list = [32, 64, 128, 256],
                 dropout_linear: float = 0.3,
                 sigma_init: float = 0.017,
                 save_images: bool = False,
                 output_dir: str = './output'):

        super(DDQAugmentedNoisyTransformerNN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self._model_name = 'DDQAugmentedNoisyTransformerNN'

        # cnn layer for feature extraction
        self.cnnex = CNNExtractor(input_shape, conv_channels, save_images, output_dir)

        # calculate the dimensionality of the feature vector
        self.num_channels = conv_channels[-1]  # number of channels from the last CNN layer
        self.seq_length = self.calculate_seq_length(input_shape)  # calculate the sequence length based on input shape

        # stack of transformer encoder layers for deep sequence processing
        self.transformer_encoders = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.num_channels, nhead=num_heads) for _ in range(num_layers)
        ])

        # Define Noisy layers for advantage value computation
        self.advantage = nn.Sequential(
            NoisyLinear(in_features=self.num_channels * self.seq_length,
                        out_features=size_linear_layers,
                        sigma_init=sigma_init),
            nn.ReLU(),
            nn.Dropout(dropout_linear),
            NoisyLinear(in_features=size_linear_layers,
                        out_features=self.num_actions,
                        sigma_init=sigma_init)
        )

        # Define Noisy layers for state value computation
        self.value = nn.Sequential(
            NoisyLinear(in_features=self.num_channels * self.seq_length,
                        out_features=size_linear_layers,
                        sigma_init=sigma_init),
            nn.ReLU(),
            nn.Dropout(dropout_linear),
            NoisyLinear(in_features=size_linear_layers,
                        out_features=1,
                        sigma_init=sigma_init)
        )

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, name: str):
        self._model_name = name

    def forward(self, x):
        # extract features using CNN
        x = self.cnnex(x)

        # Debugging-Ausgabe
        logging.debug(f"Shape after CNN: {x.shape}")

        # get batch size
        batch_size = x.size(0)

        # Calculate the correct seq_length based on the actual output shape of CNN
        seq_length = x.size(2) * x.size(3)
        logging.debug(f"Calculated seq_length: {seq_length}")

        # reshape the features to include sequence dimension
        x = x.view(batch_size, self.num_channels, self.seq_length)
        logging.debug(f"Shape after view: {x.shape}")
        x = x.permute(0, 2, 1)
        logging.debug(f"Shape after permute: {x.shape}")

        # process through each transformer encoder layer
        for encoder in self.transformer_encoders:
            x = encoder(x)
            logging.debug(f"Shape after transformer encoder: {x.shape}")

        # remove the sequence dimension post transformer
        x = x.view(batch_size, -1)
        logging.debug(f"Shape after view to remove sequence dimension: {x.shape}")

        # apply layer normalization
        x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
        logging.debug(f"Shape after layer normalization: {x.shape}")

        # compute advantage values
        advantage = self.advantage(x)
        logging.debug(f"Shape after advantage calculation: {advantage.shape}")

        # compute state value
        value = self.value(x)
        logging.debug(f"Shape after value calculation: {value.shape}")

        # combine advantage and value to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        logging.debug(f"Shape of Q-values: {q_values.shape}")

        return q_values.squeeze()

    def calculate_seq_length(self, input_shape: tuple) -> int:
        """
        Calculate the sequence length based on the input shape and the CNN architecture.

        Args:
            input_shape (tuple): The shape of the input images (channels, height, width).

        Returns:
            int: The calculated sequence length.
        """
        # Create a dummy input tensor
        dummy_input = torch.zeros((1, *input_shape))

        # Pass the dummy input through the CNN
        dummy_output = self.cnnex(dummy_input)
        logging.debug(f"Shape of dummy output: {dummy_output.shape}")

        # Calculate the sequence length based on the output shape
        seq_length = dummy_output.size(2) * dummy_output.size(3)
        logging.debug(f"Calculated seq_length in calculate_seq_length: {seq_length}")

        return seq_length