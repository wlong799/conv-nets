# coding=utf-8

"""Creates input pipeline from CIFAR-10 binary datasets.

The methods in this module build an input pipeline that parses the CIFAR-10
binaries into batches of labeled example images. All methods rely on
TensorFlow's queueing operations, so that input preprocessing and model
training/evaluation can run in parallel, ensuring the application does not
get bottlenecked due to unavailability of data. Data reader threads add data
to the queue as they read it, while threads used by the model simultaneously
dequeue the labeled examples one batch at a time.

The CIFAR-10 binary format is described here:
http://www.cs.toronto.edu/~kriz/cifar.html

More info on reading data and using queues in TensorFlow can be found here:
https://www.tensorflow.org/programmers_guide/reading_data
https://www.tensorflow.org/programmers_guide/threading_and_queues
"""
import os

import sys
import tarfile
import urllib.request

import tensorflow as tf

__all__ = ['get_input_batch']

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
TRAIN_DATA_NAMES = ['data_batch_{0:d}'.format(i) for i in range(1, 6)]
TEST_DATA_NAMES = ['test_batch']
NUM_TRAIN_EXAMPLES_PER_EPOCH = 50000
NUM_TEST_EXAMPLES_PER_EPOCH = 10000

RAW_IMAGE_DIM = [32, 32, 3]
PROCESSED_IMAGE_DIM = [24, 24, 3]

MAX_BRIGHTNESS_DELTA = 50
CONTRAST_MIN = 0.5
CONTRAST_MAX = 1.5

NUM_PREPROCESSING_THREADS = 16
MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.5


def get_input_batch(data_dir, batch_size, use_eval_data=False):
    """Obtains batch of images to use for training or testing.

    Parses the CIFAR-10 binaries located in the specified data directory
    (downloading them first if necessary), applies preprocessing steps to the
    images, and provides labeled batches of the images to be used by the CNN,
    either for training or evaluation.

    Args:
        data_dir: Path to CIFAR-10 data directory.
        batch_size: Number of examples per batch.
        use_eval_data: bool. Whether to use test instead of training data.
    Returns:
        image_batch: 4D Tensor of [batch_size, PROCESSED_IMAGE_DIM*]. Batch
        of processed images to feed into the model.
        label_batch: 1D int32 Tensor of [batch_size]. Labels in the range
        0-9, corresponding to the class of the images in image_batch.
    """
    download_and_extract_binaries(data_dir)
    filenames = TEST_DATA_NAMES if use_eval_data else TRAIN_DATA_NAMES
    filenames = [os.path.join(data_dir, filename) for filename in filenames]
    filename_queue = tf.train.string_input_producer(filenames)

    image, label = parse_cifar10_binary(filename_queue)
    processed_image = process_image(image, distort=use_eval_data)
    label.set_shape([1])
    processed_image.set_shape(PROCESSED_IMAGE_DIM)

    num_examples_per_epoch = NUM_TEST_EXAMPLES_PER_EPOCH if use_eval_data \
        else NUM_TRAIN_EXAMPLES_PER_EPOCH
    min_examples_after_dequeue = num_examples_per_epoch * \
                                 MIN_FRACTION_OF_EXAMPLES_IN_QUEUE
    capacity = min_examples_after_dequeue + (NUM_PREPROCESSING_THREADS + 2) \
                                            * batch_size
    if use_eval_data:
        image_batch, label_batch = tf.train.batch(
            [processed_image, label], batch_size, NUM_PREPROCESSING_THREADS,
            capacity)
    else:
        image_batch, label_batch = tf.train.shuffle_batch(
            [processed_image, label], batch_size, capacity,
            min_examples_after_dequeue, NUM_PREPROCESSING_THREADS)
    return image_batch, label_batch





def parse_cifar10_binary(filename_queue):
    """Parses labeled example from binary CIFAR-10 file.

    Args:
        filename_queue: Queue of filenames to parse binaries from.

    Returns:
        image: 3D uint8 Tensor of RAW_IMAGE_DIM. The input image.
        label: int32 Tensor with value in range [0, 9]. The input class.
    """
    num_label_bytes = 1
    height, width, depth = RAW_IMAGE_DIM
    num_image_bytes = height * width * depth
    reader = tf.FixedLengthRecordReader(num_label_bytes + num_image_bytes)
    _, raw_string = reader.read(filename_queue)
    raw_bytes = tf.decode_raw(raw_string, tf.uint8)
    label = tf.cast(tf.strided_slice(raw_bytes, [0], [num_label_bytes]),
                    tf.int32)
    depth_first_image = tf.reshape(
        tf.strided_slice(raw_bytes, [num_label_bytes],
                         [num_label_bytes + num_image_bytes]),
        [depth, height, width])
    image = tf.transpose(depth_first_image, [1, 2, 0])
    return image, label


def process_image(image, distort):
    """Applies preprocessing steps to the raw input image.

    Args:
        image: 3D uint8 Tensor of RAW_IMAGE_DIM. The image to be processed.
        distort: bool. Whether random distortion steps should be applied to
        the image (i.e. for better regularization during training).
    Returns:
        processed_image: 3D float32 tensor of PROCESSED_IMAGE_DIM.
    """
    # TODO: Determine proper distortions/values for distortions.
    processed_image = tf.cast(image, tf.float32)
    if distort:
        processed_image = tf.random_crop(processed_image, PROCESSED_IMAGE_DIM)
        processed_image = tf.image.random_flip_left_right(processed_image)
        processed_image = tf.image.random_brightness(processed_image,
                                                     MAX_BRIGHTNESS_DELTA)
        processed_image = tf.image.random_contrast(processed_image,
                                                   CONTRAST_MIN, CONTRAST_MAX)
    else:
        processed_image = tf.image.resize_image_with_crop_or_pad(
            processed_image, PROCESSED_IMAGE_DIM[0], PROCESSED_IMAGE_DIM[1])
    processed_image = tf.image.per_image_standardization(processed_image)
    return processed_image
