# coding=utf-8

"""Creates input pipeline from CIFAR-10 binary datasets.

The methods in this module build an input pipeline that parses the CIFAR-10
binaries into batches of labeled example images. All methods rely on
TensorFlow's queueing operations, so that input preprocessing and model
training/evaluation can run in parallel, ensuring the application does not
get bottlenecked due to unavailability of data. Data reader threads add data
to the queue as they read it, while threads used by the model simultaneously
dequeue the labeled examples one batch at a time.
"""
import functools
import random

import tensorflow as tf

from .dataset import Dataset


def get_minibatch(dataset: Dataset, phase, batch_size, distort_images,
                  min_example_fraction, num_preprocessing_threads,
                  data_format):
    """Obtains batch of images to use for training or testing.

    Parses the proper subset of TFRecords files from the supplied dataset,
    applies preprocessing steps to the images, and returns the labeled
    examples one batch at a time.

    Args:
        dataset: Dataset to return examples from.
        phase: 'train', 'valid' or 'test'. Which subset of data to use.
        batch_size: int. Number of examples per batch.
        distort_images: bool. Whether to distort images during training phase.
        min_example_fraction: float. Fraction of examples in buffer.
        num_preprocessing_threads: int. Number threads to use for processing.
        data_format: 'NCHW' or 'NHWC'

    Returns:
        image_batch: 4D float32 Tensor [batch_size, image_height, image_width,
                     image_channels] (if NHWC format). Batch of processed
                     images to feed into the model.

        label_batch: 1D int32 Tensor [batch_size]. Labels corresponding to the
                     class of the images in image_batch.
    """
    with tf.name_scope('preprocess_{}_batch'.format(phase)):
        # Get queue of TFRecord files from proper subset
        filename_queue = dataset.get_filename_queue(phase)

        # Parse an image and apply preprocessing steps
        image, label, bbox = _parse_example(filename_queue)
        use_distortions = distort_images and phase == 'train'
        image_height, image_width, image_channels = dataset.image_shape
        processed_image = process_image(image, image_height, image_width,
                                        image_channels, use_distortions)

        # Add image summaries
        tf.summary.image('original', tf.expand_dims(image, 0), 1)
        tf.summary.image('processed', tf.expand_dims(processed_image, 0), 1)

        # Set up queue of example batches
        min_buffer_size = int(dataset.examples_per_epoch(phase) *
                              min_example_fraction)
        capacity = min_buffer_size + num_preprocessing_threads * batch_size
        image_batch, label_batch = tf.train.shuffle_batch(
            [processed_image, label], batch_size, capacity,
            min_buffer_size, num_preprocessing_threads)
        if data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, [0, 3, 1, 2])
        return image_batch, label_batch


def _parse_example(filename_queue):
    """Parses labeled example from filename queue of TFRecords files.

    Args:
        filename_queue: Queue of filenames to parse TFRecords Example from.

    Returns:
        image: 3D uint8 Tensor [height, width, channels]. Image in RGB.
        label: int32 Tensor. The input class.
    """
    # TODO: Do something with all the other features that are stored
    with tf.name_scope('parse_example'):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'class/label': tf.FixedLenFeature([], tf.int64),
            'bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        })
        image = tf.image.decode_jpeg(features['image/encoded'])
        label = tf.cast(features['class/label'], tf.int32)

        xmin = tf.expand_dims(features['bbox/xmin'].values, 0)
        xmax = tf.expand_dims(features['bbox/xmax'].values, 0)
        ymin = tf.expand_dims(features['bbox/ymin'].values, 0)
        ymax = tf.expand_dims(features['bbox/ymax'].values, 0)
        bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
        return image, label, bbox


def process_image(image, image_height, image_width, image_channels,
                  use_distortions):
    """Applies preprocessing steps to the raw input image.

    Args:
        image: 3D uint8 Tensor representing image. Either grayscale or RGB.
        image_height: int. Height of processed image.
        image_width: int. Width of processed image.
        image_channels: int. Number of channels in image.
        use_distortions: bool. Whether random distortion steps should be used.
    Returns:
        processed_image: float32 image Tensor [image_height, image_width, 3].
    """
    # TODO: Consider making other resizing options available instead
    with tf.name_scope('process_image'):
        processed_image = tf.image.convert_image_dtype(image, tf.float32)
        if use_distortions:
            with tf.name_scope('distort_image'):
                processed_image = tf.random_crop(
                    processed_image,
                    [image_height, image_width, image_channels])
                processed_image = _distort_color(processed_image)
        else:
            processed_image = tf.image.resize_image_with_crop_or_pad(
                processed_image, image_height, image_width)
        processed_image.set_shape([image_height, image_width, image_channels])
        return tf.image.per_image_standardization(processed_image)


def _distort_color(image):
    """Performs random color distortions in a random order."""
    with tf.name_scope('distort_color'):
        distort_funcs = \
            [lambda img: tf.image.random_brightness(img, 32.0 / 255),
             lambda img: tf.image.random_contrast(img, 0.5, 1.5),
             lambda img: tf.image.random_hue(img, 0.2),
             lambda img: tf.image.random_saturation(img, 0.5, 1.5)]
        random.shuffle(distort_funcs)
        distorted_image = \
            functools.reduce(lambda res, f: f(res), distort_funcs, image)
    return distorted_image