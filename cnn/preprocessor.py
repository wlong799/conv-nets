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
import os
import random

import tensorflow as tf

import cnn


def get_minibatch(model_config: cnn.config.ModelConfig):
    """Obtains batch of images to use for training or testing.

    Parses the proper TFRecords file located in the data directory, applies
    preprocessing steps to the images, and returns the labeled examples one
    batch at a time.

    Expects files in the format *phase*.tfrecords, where phase is the current
    phase as specified by the configuration.

    Args:
        model_config: Model configuration.

    Returns:
        image_batch: 4D float32 Tensor [batch_size, image_height,
                     image_width, 3] (if NHWC format). Batch of processed
                     images to feed into the model.
        label_batch: 1D int32 Tensor [batch_size]. Labels in the range
                     0-9, corresponding to the class of the images in
                     image_batch.
    """
    if model_config.phase not in ['train', 'valid', 'test']:
        raise ValueError("Phase must be one of 'train', 'valid', or 'test'")
    with tf.name_scope('{}_batch_preprocessing'.format(model_config.phase)):
        # Add proper TFRecord files to queue
        pattern = '*{}*.tfrecords'.format(model_config.phase)
        pattern = os.path.join(model_config.data_dir, pattern)
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(pattern))

        # Parse an image and apply preprocessing steps
        image, label = _parse_cifar10_example(filename_queue)
        use_distortions = (model_config.distort_images and
                           model_config.phase == 'train')
        processed_image = process_image(
            image, model_config.image_height, model_config.image_width,
            model_config.image_channels, use_distortions)

        # Add image summaries
        tf.summary.image('original', tf.expand_dims(image, 0), 1)
        tf.summary.image('processed', tf.expand_dims(processed_image, 0), 1)

        # Set up queue of example batches
        capacity = (model_config.min_buffer_size +
                    model_config.num_preprocessing_threads *
                    model_config.batch_size)
        if model_config.phase == 'train':
            image_batch, label_batch = tf.train.shuffle_batch(
                [processed_image, label], model_config.batch_size, capacity,
                model_config.min_buffer_size,
                model_config.num_preprocessing_threads)
        else:
            image_batch, label_batch = tf.train.batch(
                [processed_image, label], model_config.batch_size,
                model_config.num_preprocessing_threads, capacity)
        if model_config.data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, [0, 3, 1, 2])
        return image_batch, label_batch


def _parse_cifar10_example(filename_queue):
    """Parses labeled example from filename queue of TFRecords files.

    Args:
        filename_queue: Queue of filenames to parse TFRecords Example from.

    Returns:
        image: 3D uint8 Tensor [height, width, channels]. Image in RGB.
        label: int32 Tensor with value in range [0, 9]. The input class.
    """
    with tf.name_scope('example_parsing'):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'encoded_image': tf.FixedLenFeature([], tf.string)
        })
        label = tf.cast(features['label'], tf.int32)
        image = tf.image.decode_png(features['encoded_image'], 3)
        return image, label


def process_image(image, image_height, image_width, image_channels,
                  use_distortions):
    """Applies processing steps to the raw input image.

    Args:
        image: 3D uint8 Tensor representing image. Either grayscale or RGB.
        image_height: int. Height of processed image.
        image_width: int. Width of processed image.
        image_channels: int. Number of channels in image.
        use_distortions: bool. Whether random distortion steps should be applied.
    Returns:
        processed_image: float32 image Tensor [image_height, image_width, 3].
    """
    # TODO: Consider making other resizing options available instead
    with tf.name_scope('image_processing'):
        processed_image = tf.image.convert_image_dtype(image, tf.float32)
        if use_distortions:
            with tf.name_scope('image_distortion'):
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
    distort_funcs = [lambda img: tf.image.random_brightness(img, 32.0 / 255),
                     lambda img: tf.image.random_contrast(img, 0.5, 1.5),
                     lambda img: tf.image.random_hue(img, 0.2),
                     lambda img: tf.image.random_saturation(img, 0.5, 1.5)]
    random.shuffle(distort_funcs)
    return functools.reduce(lambda res, f: f(res), distort_funcs, image)
