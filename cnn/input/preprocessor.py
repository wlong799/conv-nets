# coding=utf-8

"""Creates input pipeline from CIFAR-10 binary datasets.

The methods in this module build an input pipeline using TensorFlow's
queueing operations, so that input preprocessing and model
training/evaluation can run in parallel, ensuring the application does not
get bottlenecked due to unavailability of data."""
import functools
import random

import tensorflow as tf

from .datasets import Dataset


def get_minibatch(dataset: Dataset, phase, batch_size, distort_images,
                  min_example_fraction, num_preprocessing_threads,
                  data_format):
    """Reads and processes batch of labeled images from TFRecords file.

    Args:
        dataset: Dataset to return examples from.
        phase: 'train', 'valid' or 'test'. Which subset of data to use.
        batch_size: int. Number of examples per batch.
        distort_images: bool. Whether to distort images during training phase.
        min_example_fraction: float. Fraction of examples in buffer.
        num_preprocessing_threads: int. Number threads to use for processing.
        data_format: 'NCHW' or 'NHWC'.

    Returns:
        image_batch: 4D float32 Tensor [batch_size, image_height, image_width,
                     image_channels] (if NHWC format). Batch of processed
                     images to feed into the model.
        label_batch: 1D int32 Tensor [batch_size]. Labels corresponding to the
                     class of the images in image_batch.
        text_batch: 1D string Tensor [batch_size]. Text name of the class for
                    each image in image_batch.
    """
    with tf.name_scope('{}_batch_preprocessing'.format(phase)):
        image, label, text = dataset.read_example(phase)
        use_distortions = distort_images and phase == 'train'
        processed_image = _process_image(image, dataset.image_shape,
                                         use_distortions)

        tf.summary.image('original', tf.expand_dims(image, 0), 1)
        tf.summary.image('processed', tf.expand_dims(processed_image, 0), 1)

        min_buffer_size = int(dataset.examples_per_epoch(phase) *
                              min_example_fraction)
        safety_buffer_size = num_preprocessing_threads * batch_size
        queue_capacity = min_buffer_size + safety_buffer_size
        image_batch, label_batch, text_batch = tf.train.shuffle_batch(
            [processed_image, label, text], batch_size, queue_capacity,
            min_buffer_size, num_preprocessing_threads)
        if data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, [0, 3, 1, 2])
        return image_batch, label_batch, text_batch


def _process_image(image, image_shape, use_distortions):
    """Applies preprocessing steps to the raw input image, by cropping it to
    the specified size, applying color distortions if specified,
    and normalizing the values."""
    with tf.name_scope('image_processing'):
        processed_image = tf.image.convert_image_dtype(image, tf.float32)
        if use_distortions:
            with tf.name_scope('image_distortions'):
                processed_image = tf.random_crop(processed_image, image_shape)
                processed_image = _distort_color(processed_image)
        else:
            height, width = image_shape[:2]
            processed_image = tf.image.resize_image_with_crop_or_pad(
                processed_image, height, width)
        processed_image.set_shape(image_shape)
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
