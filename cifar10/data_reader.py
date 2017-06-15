# coding=utf-8

"""Creates input pipeline from CIFAR-10 binary datasets.

The methods in this module decode the CIFAR-10 binaries located in the
specified data directory (or downloading them first if necessary), apply
preprocessing steps to the images, and provide labeled batches of the images to
be used by the CNN, either for training or evaluation.

TensorFlow makes it possible to read data from files using multiple threads,
separate from the threads used for running the model. This helps to ensure the
application is not bottlenecked due to running out of available data. As
the data is read, it is stored in a queue. Labeled examples are then
dequeued one batch at a time when requested by the model.

The CIFAR-10 binary format is described here:
http://www.cs.toronto.edu/~kriz/cifar.html

More information on reading data in TensorFlow can be found here:
https://www.tensorflow.org/programmers_guide/reading_data
"""
__all__ = ['download_and_extract_binaries', 'get_input_batch']

# TODO: Update with proper values
# TODO: Move to config file?
DATA_URL = 'asdf'
TRAIN_DATA_PREFIX = 'asdf'
TEST_DATA_PREFIX = 'asdf'
IMAGE_SIZE = [24, 24]


def download_and_extract_binaries(data_dir):
    """Downloads and extracts zipped CIFAR-10 binaries from web.

    Args:
        data_dir: Path to CIFAR-10 data directory.
    """
    # TODO: Write function


def decode_binaries(filename_queue):
    """Returns a single labeled image from the binary file format.

    Args:
        filename_queue: FIFO queue of filenames to pull binaries from.
    """
    # TODO: Write function


def process_image(image, distort):
    """Applies preprocessing steps to the raw input image.

    Args:
        image: Input image.
        distort: bool, whether random distortion steps should be applied to
        the image (i.e. for better regularization during training).
    """
    # TODO: Write function


def get_input_batch(data_dir, use_eval_data=False):
    """Gets batch of images for training or testing.

    Args:
        data_dir: Path to CIFAR-10 data directory.
        use_eval_data: bool, whether or not to use test data.
    """
    # TODO: Write function
