import cv2
import numpy as np


class FrameProcessor:
    """
    This class handles the preprocessing of frames from the environment.
    It includes methods to preprocess individual frames and stack them
    into a single input tensor suitable for use in deep learning models.

    Attributes:
        environment (np.ndarray): The game environment providing the frames to preprocess.
    """

    def __init__(self) -> None:
        """
        Initializes the Preprocessing class with the game environment.

        Args:

        """
        self.stacked_frames = np.empty((0, 0))

    def preprocess(self, stacked_frames: np.ndarray, env_state: np.ndarray, exclude: tuple[int, int, int, int],
                   output: int, is_new: bool = False) -> np.ndarray:
        """
        Processes the current frame from the environment and stacks it to create an input tensor.

        This method first preprocesses a single frame from the environment by converting it to grayscale,
        cropping it, normalizing the pixel values, and resizing it. It then stacks the preprocessed frame
        together with previous frames to form a multi-channel input tensor, providing the agent with a
        sense of motion.

        Args:
            stacked_frames (np.ndarray): Array of stacked frames (four channels).
            env_state (tuple): The current state of the environment, typically containing the frame data.
            exclude (tuple[int, int, int, int]): The section to be cropped (UP, RIGHT, DOWN, LEFT).
            output (int): The size of the output image.
            is_new (bool): Flag indicating if this is the first frame in the sequence. Defaults to False.

        Returns:
            np.ndarray: The stacked frames as a single input tensor.
        """

        # Preprocess the current frame from the environment
        single_frame = self.process_image(env_state, exclude, output)

        # Stack the preprocessed frame to create a multi-channel input tensor
        stacked_frames = self.stack_frames(stacked_frames, single_frame, is_new)

        # Return the stacked frames
        return stacked_frames

    def process_image(self, env_array: np.ndarray, exclude: tuple[int, int, int, int], output: int) -> np.ndarray:
        """
        Preprocesses a single frame from the environment by performing several image processing steps.

        The method converts the image to grayscale, crops it according to the specified dimensions,
        normalizes the pixel values, and resizes it to the desired output size.

        Args:
            env_array (np.ndarray): The input frame from the environment.
            exclude (tuple[int, int, int, int]): The section to be cropped (UP, RIGHT, DOWN, LEFT).
            output (int): The desired size of the output image.

        Returns:
            np.ndarray: The preprocessed frame.
        """

        # Convert image to gray scale
        screen = cv2.cvtColor(env_array, cv2.COLOR_RGB2GRAY)

        # Crop the screen (Up:Down, Left:Right)
        screen = screen[exclude[0]:exclude[2], exclude[3]:exclude[1]]

        # Convert to float and normalize the pixel values
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

        # Resize the image to the desired output size
        screen = cv2.resize(screen, (output, output), interpolation=cv2.INTER_AREA)

        # return single screen for stack creation
        return screen

    def stack_frames(self, stacked_frames: np.ndarray, frame: np.ndarray, is_new: bool = False) -> np.ndarray:
        """
        Stacks frames together to create a single input tensor with multiple channels,
        which provides a sense of motion to the agent.

        Args:
            stacked_frames (np.ndarray): Array of stacked frames (four channels).
            frame (np.ndarray): Preprocessed frame to be added.
            is_new (bool): Flag indicating if this is the first frame in the sequence.

        Returns:
            np.ndarray: The updated stack of frames.
        """
        if is_new:
            # Initialize a new stack of frames if it's the first frame
            stacked_frames = np.stack([frame, frame, frame, frame])
        else:
            # Roll the stack to the left and add the new frame at the end
            stacked_frames = np.roll(stacked_frames, shift=-1, axis=0)
            stacked_frames[-1] = frame

        return stacked_frames
