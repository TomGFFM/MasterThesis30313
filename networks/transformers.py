import torch
import torch.nn as nn
from networks import CNNExtractor
import torch.autograd as autograd


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

    @property
    def model_name(self):
        """
        Returns the name of the model.

        Returns:
            str: The name of the model.
        """
        return 'DDQAugmentedTransformerNN'

    def __init__(self, input_shape: tuple, num_actions, num_heads=8, num_layers=6,
                 conv_channels: list = [32, 64, 128, 256], save_images: bool = False, output_dir: str = './output'):
        super(DDQAugmentedTransformerNN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

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
            nn.Linear(self.num_channels * self.seq_length, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_actions)
        )

        # Define fully connected layers for state value computation
        self.value = nn.Sequential(
            nn.Linear(self.num_channels * self.seq_length, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # extract features using CNN
        x = self.cnnex(x)

        # get batch size
        batch_size = x.size(0)

        # reshape the features to include sequence dimension
        x = x.view(batch_size, self.num_channels, self.seq_length)
        x = x.permute(0, 2, 1)

        # process through each transformer encoder layer
        for encoder in self.transformer_encoders:
            x = encoder(x)

        # remove the sequence dimension post transformer
        x = x.view(batch_size, -1)

        # apply layer normalization
        x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)

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
