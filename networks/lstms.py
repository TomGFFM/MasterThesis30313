import logging

import torch
import torch.nn as nn
from networks import CNNExtractor, CNNExtractorLean, NoisyLinear


class DDQAugmentedLSTMNN(nn.Module):
    """
    Integrates convolutional layers with LSTM layers to enhance feature detection capabilities
    and manage long-range dependencies. This hybrid model is designed for tasks requiring
    both local and global context awareness.

    Attributes:
        cnnex (CNNExtractor): Initial convolution layer to extract preliminary features from input.
        lstm_layers (nn.LSTM): LSTM layers to process sequence data.
        advantage (nn.Sequential): Fully connected layers for advantage value computation.
        value (nn.Sequential): Fully connected layers for state value computation.

    Args:
        input_shape (tuple): Shape of the input (channels, height, width).
        num_actions (int): Number of possible actions or outputs the model can generate.
        hidden_size (int): Number of features in the hidden state of the LSTM.
        num_layers (int): Number of LSTM layers.
        conv_channels (list): The number of output channels for each convolutional layer in CNNExtractor.
                              Default: [32, 64, 128, 256]
        save_images (bool): Whether to save the output images of each convolutional layer in CNNExtractor.
                            Default: False
        lean_cnn (bool): Whether to use a Lean CNN layer in CNNExtractor.
                         Default: False
        output_dir (str): The directory to save the output images from CNNExtractor.
                          Default: 'output'
    """

    def __init__(self,
                 input_shape: tuple,
                 num_actions: int,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 conv_channels: list = [32, 64, 128, 256],
                 save_images: bool = False,
                 lean_cnn: bool = False,
                 output_dir: str = './output'):

        super(DDQAugmentedLSTMNN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self._model_name = 'DDQAugmentedLSTMNN'

        # cnn layer for feature extraction
        if lean_cnn:
            self.cnnex = CNNExtractorLean(input_shape, conv_channels, save_images, output_dir)
        else:
            self.cnnex = CNNExtractor(input_shape, conv_channels, save_images, output_dir)


        # calculate the dimensionality of the feature vector
        self.num_channels = conv_channels[-1]  # number of channels from the last CNN layer
        self.seq_length = self.calculate_seq_length(input_shape)  # calculate the sequence length based on input shape

        # LSTM layers for sequence processing
        self.lstm_layers = nn.LSTM(
            input_size=self.num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Define fully connected layers for advantage value computation
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_actions)
        )

        # Define fully connected layers for state value computation
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
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

        # get batch size
        batch_size = x.size(0)

        # reshape the features to include sequence dimension
        x = x.view(batch_size, self.seq_length, self.num_channels)

        # process through LSTM layers
        x, _ = self.lstm_layers(x)

        # get the last output of the LSTM
        x = x[:, -1, :]

        # compute advantage values
        advantage = self.advantage(x)

        # compute state value
        value = self.value(x)

        # combine advantage and value to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

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

        # Calculate the sequence length based on the output shape
        seq_length = dummy_output.size(2) * dummy_output.size(3)

        return seq_length


class DDQAugmentedNoisyLSTMNN(nn.Module):
    """
    Integrates convolutional layers with LSTM layers to enhance feature detection capabilities
    and manage long-range dependencies. This hybrid model is designed for tasks requiring
    both local and global context awareness.

    Attributes:
        cnnex (CNNExtractor): Initial convolution layer to extract preliminary features from input.
        lstm_layers (nn.LSTM): LSTM layers to process sequence data.
        advantage (nn.Sequential): Fully connected layers with NoisyLinear for advantage value computation.
        value (nn.Sequential): Fully connected layers with NoisyLinear for state value computation.

    Args:
        input_shape (tuple): Shape of the input (channels, height, width).
        num_actions (int): Number of possible actions or outputs the model can generate.
        hidden_size (int): Number of features in the hidden state of the LSTM.
        num_layers (int): Number of LSTM layers.
        conv_channels (list): The number of output channels for each convolutional layer in CNNExtractor.
                              Default: [32, 64, 128, 256]
        save_images (bool): Whether to save the output images of each convolutional layer in CNNExtractor.
                            Default: False
        lean_cnn (bool): Whether to use a Lean CNN layer in CNNExtractor.
                         Default: False
        output_dir (str): The directory to save the output images from CNNExtractor.
                          Default: 'output'
    """

    def __init__(self,
                 input_shape: tuple,
                 num_actions: int,
                 hidden_size: int = 256,
                 num_layers: int = 6,
                 size_linear_layers: int = 512,
                 conv_channels: list = [32, 64, 128, 256],
                 dropout_linear: float = 0.3,
                 sigma_init: float = 0.017,
                 save_images: bool = False,
                 lean_cnn: bool = False,
                 output_dir: str = './output'):

        super(DDQAugmentedNoisyLSTMNN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self._model_name = 'DDQAugmentedNoisyLSTMNN'

        # cnn layer for feature extraction
        if lean_cnn:
            self.cnnex = CNNExtractorLean(input_shape, conv_channels, save_images, output_dir)
        else:
            self.cnnex = CNNExtractor(input_shape, conv_channels, save_images, output_dir)

        # calculate the dimensionality of the feature vector
        self.num_channels = conv_channels[-1]  # number of channels from the last CNN layer
        self.seq_length = self.calculate_seq_length(input_shape)  # calculate the sequence length based on input shape

        # Init layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=True)

        # LSTM layers for sequence processing
        self.lstm_layers = nn.LSTM(
            input_size=self.num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Define Noisy layers for advantage value computation
        self.advantage = nn.Sequential(
            NoisyLinear(in_features=hidden_size,
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
            NoisyLinear(in_features=hidden_size,
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

        # get batch size
        batch_size = x.size(0)

        # Calculate the correct seq_length based on the actual output shape of CNN
        seq_length = x.size(2) * x.size(3)
        logging.debug(f"Calculated seq_length: {seq_length}")

        # reshape the features to include sequence dimension
        # x = x.view(batch_size, self.seq_length, self.num_channels)

        # reshape the features to include sequence dimension
        x = x.view(batch_size, self.num_channels, self.seq_length)
        logging.debug(f"Shape after view: {x.shape}")
        x = x.permute(0, 2, 1)
        logging.debug(f"Shape after permute: {x.shape}")

        # process through LSTM layers
        x, _ = self.lstm_layers(x)

        # get the last output of the LSTM
        x = x[:, -1, :]

        # apply layer normalization
        x = self.layer_norm(x)
        logging.debug(f"Shape after layer normalization: {x.shape}")

        # compute advantage values
        advantage = self.advantage(x)

        # compute state value
        value = self.value(x)

        # combine advantage and value to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

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

        # Calculate the sequence length based on the output shape
        seq_length = dummy_output.size(2) * dummy_output.size(3)

        return seq_length