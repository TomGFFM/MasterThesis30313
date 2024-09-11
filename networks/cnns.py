import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import torchvision.utils as vutils
from PIL import Image


class DQNetworkCNN(nn.Module):
    """
    Deep Q-Network (DQN) Convolutional Neural Network (CNN) for processing
    input frames from an environment and outputting Q-values for each action.

    This network uses a series of convolutional layers to extract features from
    the input frames, followed by fully connected layers to produce the final
    Q-values.

    Attributes:
        input_shape (tuple): The shape of the input frames (channels, height, width).
        num_actions (int): The number of possible actions in the environment.
        conv_channels (list): The number of output channels for each convolutional layer.
        features (nn.Sequential): Convolutional layers for feature extraction.
        fc (nn.Sequential): Fully connected layers for Q-value computation.
    """
    def __init__(self, input_shape: tuple, num_actions: int, conv_channels: list = [32, 64, 64]) -> None:
        """
        Initializes the DQNetworkCNN with the given input shape, number of actions, and convolutional channels.

        Args:
            input_shape (tuple): The shape of the input frames (channels, height, width).
            num_actions (int): The number of possible actions in the environment.
            conv_channels (list): The number of output channels for each convolutional layer.
                                  Default: [32, 64, 64]
        """
        super(DQNetworkCNN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.conv_channels = conv_channels

        # Define convolutional layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], conv_channels[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Define fully connected layers for Q-value computation
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    @property
    def model_name(self):
        return 'DQNetworkCNN'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network to compute Q-values for the input frames.

        Args:
            x (torch.Tensor): The input tensor representing a batch of frames.

        Returns:
            torch.Tensor: The output tensor representing Q-values for each action.
        """
        # Extract features using convolutional layers
        x = self.features(x)

        # Flatten the output of convolutional layers
        x = x.view(x.size(0), -1)

        # Compute Q-values using fully connected layers
        x = self.fc(x)

        return x

    def feature_size(self) -> int:
        """
        Computes the size of the feature map after the convolutional layers.

        This method runs a forward pass with a dummy input to determine the
        size of the feature map, which is needed to define the input size
        of the first fully connected layer.

        Returns:
            int: The size of the flattened feature map.
        """
        # Create a dummy input tensor with the same shape as the input frames
        dummy_input = autograd.Variable(torch.zeros(1, *self.input_shape))

        # Pass the dummy input through the convolutional layers
        features = self.features(dummy_input)

        # Flatten the output and compute the size
        return features.view(1, -1).size(1)


class DDQNetworkCNN(nn.Module):
    """
    Double Deep Q-Network (DDQN) Convolutional Neural Network (CNN) for processing
    input frames from an environment and outputting Q-values for each action.

    This network uses a series of convolutional layers to extract features from
    the input frames, followed by separate advantage and value streams to compute
    the Q-values using the dueling architecture.

    Attributes:
        input_shape (tuple): The shape of the input frames (channels, height, width).
        num_actions (int): The number of possible actions in the environment.
        conv_channels (list): The number of output channels for each convolutional layer.
        features (nn.Sequential): Convolutional layers for feature extraction.
        advantage (nn.Sequential): Fully connected layers for advantage value computation.
        value (nn.Sequential): Fully connected layers for state value computation.
    """

    def __init__(self, input_shape: tuple, num_actions: int, conv_channels: list = [32, 64, 64]) -> None:
        """
        Initializes the DDQNetworkCNN with the given input shape, number of actions, and convolutional channels.

        Args:
            input_shape (tuple): The shape of the input frames (channels, height, width).
            num_actions (int): The number of possible actions in the environment.
            conv_channels (list): The number of output channels for each convolutional layer.
                                  Default: [32, 64, 64]
        """
        super(DDQNetworkCNN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.conv_channels = conv_channels

        # Define convolutional layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], conv_channels[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Define fully connected layers for advantage value computation
        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        # Define fully connected layers for state value computation
        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    @property
    def model_name(self):
        return 'DDQNetworkCNN'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network to compute Q-values for the input frames
        using the dueling architecture.

        Args:
            x (torch.Tensor): The input tensor representing a batch of frames.

        Returns:
            torch.Tensor: The output tensor representing Q-values for each action.
        """
        # Extract features using convolutional layers
        x = self.features(x)

        # Flatten the output of convolutional layers
        x = x.view(x.size(0), -1)

        # Compute advantage values
        advantage = self.advantage(x)

        # Compute state value
        value = self.value(x)

        # Combine advantage and value to get Q-values
        return value + advantage - advantage.mean()

    def feature_size(self) -> int:
        """
        Computes the size of the feature map after the convolutional layers.

        This method runs a forward pass with a dummy input to determine the
        size of the feature map, which is needed to define the input size
        of the fully connected layers in the advantage and value streams.

        Returns:
            int: The size of the flattened feature map.
        """
        # Create a dummy input tensor with the same shape as the input frames
        dummy_input = autograd.Variable(torch.zeros(1, *self.input_shape))

        # Pass the dummy input through the convolutional layers
        features = self.features(dummy_input)

        # Flatten the output and compute the size
        return features.view(1, -1).size(1)


class CNNExtractor(nn.Module):
    """
    A convolutional neural network (CNN) for extracting features from input images.

    Args:
        input_shape (tuple): The shape of the input images (channels, height, width).
        conv_channels (list): The number of output channels for each convolutional layer.
                              Default: [32, 64, 64]
        save_images (bool): Whether to save the output images of each convolutional layer.
                            Default: False
        output_dir (str): The directory to save the output images.
                          Default: 'output'

    Attributes:
        input_shape (tuple): The shape of the input images (channels, height, width).
        conv_channels (list): The number of output channels for each convolutional layer.
        save_images (bool): Whether to save the output images of each convolutional layer.
        output_dir (str): The directory to save the output images.
        features (nn.Sequential): The sequence of convolutional layers for feature extraction.
    """

    def __init__(self, input_shape: tuple, conv_channels: list = [32, 64, 128, 256],
                 save_images: bool = False, output_dir: str = './output') -> None:
        """
        Initializes the CNNExtractor with the given input shape, convolutional channels,
        save_images flag, and output directory.

        Args:
            input_shape (tuple): The shape of the input images (channels, height, width).
            conv_channels (list): The number of output channels for each convolutional layer.
                                  Default: [32, 64, 128, 256]
            save_images (bool): Whether to save the output images of each convolutional layer.
                                Default: False
            output_dir (str): The directory to save the output images.
                              Default: 'output'
        """
        super(CNNExtractor, self).__init__()
        self.input_shape = input_shape
        self.conv_channels = conv_channels
        self.save_images = save_images
        self.output_dir = output_dir + '/model_extractions'
        self._model_name = 'CNNExtractor'

        # Define convolutional layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], conv_channels[0], kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(conv_channels[0]),
            nn.Dropout(0.2),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(conv_channels[1]),
            nn.Dropout(0.2),
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_channels[2]),
            nn.Dropout(0.2),
            nn.Conv2d(conv_channels[2], conv_channels[3], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_channels[3]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, name: str):
        self._model_name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the CNN extractor and saves the output images
        of each convolutional layer if save_images is set to True.

        Args:
            x (torch.Tensor): The input tensor representing a batch of images.

        Returns:
            torch.Tensor: The extracted features from the input images.
        """
        # Extract features using convolutional layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            if self.save_images and isinstance(layer, nn.Conv2d):
                self.save_output_image(x, i)
        return x

    def save_output_image(self, tensor: torch.Tensor, layer_index: int, save_every_x_images: int = 3) -> None:
        """
        Saves the output images of a convolutional layer for a batch of samples.

        Args:
            tensor (torch.Tensor): The output tensor of the convolutional layer with shape (batch_size, channels, height, width).
            layer_index (int): The index of the convolutional layer.
            save_every_x_images (int): Saves every x images for each convolutional layer.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize a counter for each layer
        if not hasattr(self, 'layer_counters'):
            self.layer_counters = {}

        # Get the counter for the current layer
        if layer_index not in self.layer_counters:
            self.layer_counters[layer_index] = 0

        # Iterate over each sample in the batch
        for sample_index in range(tensor.size(0)):
            # Increment the counter for the current layer
            self.layer_counters[layer_index] += 1

            # Save the image only if the counter is a multiple of x
            if self.layer_counters[layer_index] % save_every_x_images != 0:
                continue

            # Get the tensor for the current sample
            sample_tensor = tensor[sample_index]

            # Convert the sample tensor to a numpy array
            ndarr = sample_tensor.squeeze().detach().cpu().numpy()

            # Check if the array has the expected number of dimensions
            if ndarr.ndim == 3:
                # Transpose the array to have the channel dimension last
                ndarr = np.transpose(ndarr, (1, 2, 0))
            elif ndarr.ndim == 2:
                # Add a channel dimension if the array is 2D
                ndarr = np.expand_dims(ndarr, axis=-1)
            else:
                raise ValueError(f"Unexpected number of dimensions: {ndarr.ndim}")

            # Normalize the array to the range [0, 255]
            ndarr = (ndarr - ndarr.min()) / (ndarr.max() - ndarr.min()) * 255

            # Convert the array to uint8
            ndarr = ndarr.astype('uint8')

            # Convert the array to RGB format
            if ndarr.shape[2] == 1:
                ndarr = np.repeat(ndarr, 3, axis=2)
            else:
                ndarr = ndarr[:, :, :3]

            # Create a unique filename for each sample in the batch
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_layer_{layer_index}_sample_{sample_index}.png"
            filepath = os.path.join(self.output_dir, filename)

            # Create an image from the array and save it
            image = Image.fromarray(ndarr)
            image.save(filepath)


class CNNExtractorLean(nn.Module):
    """
    A convolutional neural network (CNN) for extracting features from input images.

    Args:
        input_shape (tuple): The shape of the input images (channels, height, width).
        conv_channels (list): The number of output channels for each convolutional layer.
                              Default: [32, 64, 64]
        save_images (bool): Whether to save the output images of each convolutional layer.
                            Default: False
        output_dir (str): The directory to save the output images.
                          Default: 'output'

    Attributes:
        input_shape (tuple): The shape of the input images (channels, height, width).
        conv_channels (list): The number of output channels for each convolutional layer.
        save_images (bool): Whether to save the output images of each convolutional layer.
        output_dir (str): The directory to save the output images.
        features (nn.Sequential): The sequence of convolutional layers for feature extraction.
    """

    def __init__(self, input_shape: tuple, conv_channels: list = [32, 64, 128, 256],
                 save_images: bool = False, output_dir: str = './output') -> None:
        """
        Initializes the CNNExtractor with the given input shape, convolutional channels,
        save_images flag, and output directory.

        Args:
            input_shape (tuple): The shape of the input images (channels, height, width).
            conv_channels (list): The number of output channels for each convolutional layer.
                                  Default: [32, 64, 128, 256]
            save_images (bool): Whether to save the output images of each convolutional layer.
                                Default: False
            output_dir (str): The directory to save the output images.
                              Default: 'output'
        """
        super(CNNExtractorLean, self).__init__()
        self.input_shape = input_shape
        self.conv_channels = conv_channels
        self.save_images = save_images
        self.output_dir = output_dir + '/model_extractions'
        self._model_name = 'CNNExtractor'

        # Define convolutional layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], conv_channels[0], kernel_size=1, stride=1),
            nn.Conv2d(conv_channels[0], conv_channels[3], kernel_size=2, stride=1),
            nn.BatchNorm2d(conv_channels[3]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, name: str):
        self._model_name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the CNN extractor and saves the output images
        of each convolutional layer if save_images is set to True.

        Args:
            x (torch.Tensor): The input tensor representing a batch of images.

        Returns:
            torch.Tensor: The extracted features from the input images.
        """
        # Extract features using convolutional layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            if self.save_images and isinstance(layer, (nn.MaxPool2d)): #nn.Conv2d, nn.ReLU, nn.BatchNorm2d,
                self.save_output_image(x, i)
        return x

    def save_output_image(self, tensor: torch.Tensor, layer_index: int, save_every_x_images: int = 3) -> None:
        """
        Saves the output images of a convolutional layer for a batch of samples.

        Args:
            tensor (torch.Tensor): The output tensor of the convolutional layer with shape (batch_size, channels, height, width).
            layer_index (int): The index of the convolutional layer.
            save_every_x_images (int): Saves every x images for each convolutional layer.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize a counter for each layer
        if not hasattr(self, 'layer_counters'):
            self.layer_counters = {}

        # Get the counter for the current layer
        if layer_index not in self.layer_counters:
            self.layer_counters[layer_index] = 0

        # Iterate over each sample in the batch
        for sample_index in range(tensor.size(0)):
            # Increment the counter for the current layer
            self.layer_counters[layer_index] += 1

            # Save the image only if the counter is a multiple of x
            if self.layer_counters[layer_index] % save_every_x_images != 0:
                continue

            # Get the tensor for the current sample
            sample_tensor = tensor[sample_index]

            # Convert the sample tensor to a numpy array
            ndarr = sample_tensor.squeeze().detach().cpu().numpy()

            # Check if the array has the expected number of dimensions
            if ndarr.ndim == 3:
                # Transpose the array to have the channel dimension last
                ndarr = np.transpose(ndarr, (1, 2, 0))
            elif ndarr.ndim == 2:
                # Add a channel dimension if the array is 2D
                ndarr = np.expand_dims(ndarr, axis=-1)
            else:
                raise ValueError(f"Unexpected number of dimensions: {ndarr.ndim}")

            # Normalize the array to the range [0, 255]
            min_val = ndarr.min()
            max_val = ndarr.max()

            # Check if max_val and min_val are equal to avoid division by zero
            if max_val != min_val:
                ndarr = (ndarr - min_val) / (max_val - min_val) * 255
            else:
                ndarr = np.zeros_like(ndarr)  # Set to zeros if all values are the same

            # Convert the array to uint8
            ndarr = ndarr.astype('uint8')

            # Convert the array to RGB format
            if ndarr.shape[2] == 1:
                ndarr = np.repeat(ndarr, 3, axis=2)
            else:
                ndarr = ndarr[:, :, :3]

            # Create a unique filename for each sample in the batch
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_layer_{layer_index}_sample_{sample_index}.png"
            filepath = os.path.join(self.output_dir, filename)

            # Create an image from the array and save it
            image = Image.fromarray(ndarr)
            image.save(filepath)

