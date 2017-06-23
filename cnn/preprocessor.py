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


def get_minibatch(data_dir, batch_size, image_height=None,
                  image_width=None, phase='train', distort_images=True,
                  data_format='NHWC', num_threads=32, min_buffer_size=10000):
    """Obtains batch of images to use for training or testing.

    Parses the proper TFRecords file located in the data directory, applies
    preprocessing steps to the images, and returns the labeled examples one
    batch at a time.

    Expects files in the format *phase*.tfrecords, where phase is either train,
    test, or valid. Will not load the correct file if this naming convention is
    not used.

    Args:
        data_dir: Directory containing the TFRecords files. See above for
                  required naming convention of files.
        batch_size: int. Number of examples per batch.
        image_height: int. Height of processed images, or None for no resize.
        image_width: int. Width of processed images, or None for no resize.
        phase: Either 'train', 'test', or 'valid'.
        distort_images: bool. If true, and if training phase, images will be
                        distorted as a form of data augmentation.
        data_format: 'NHWC' or 'NCHW' Specifies data format for the processed
                     images.
        num_threads: Number of parallel threads processing images.
        min_buffer_size: Minimum number of examples to be placed in queue (
                         to ensure decent mixing).

    Returns:
        image_batch: 4D float32 Tensor [batch_size, image_height,
                     image_width, 3] (if NHWC format). Batch of processed
                     images to feed into the model.
        label_batch: 1D int32 Tensor [batch_size]. Labels in the range
                     0-9, corresponding to the class of the images in
                     image_batch.
    """
    if phase not in ['train', 'valid', 'test']:
        raise ValueError('Phase must be one of "train", "valid", or "test"')
    with tf.name_scope('{}_batch_preprocessing'.format(phase)):
        # Add proper TFRecord files to queue
        pattern = '*{}*.tfrecords'.format(phase)
        pattern = os.path.join(data_dir, pattern)
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(pattern))

        # Parse an image and apply preprocessing steps
        image, label = parse_cifar10_example(filename_queue)
        distort = distort_images and phase == 'train'
        if not image_height or not image_width:
            image_height, image_width = _get_dims(image)
        print(image_height, image_width)
        processed_image = process_image(image, image_height, image_width,
                                        distort)
        normalized_image = tf.image.per_image_standardization(processed_image)

        # Add summaries
        tf.summary.image('original_image', tf.expand_dims(image, 0))
        tf.summary.image('processed_image', tf.expand_dims(processed_image, 0))

        # Set up queue of example batches
        capacity = min_buffer_size + num_threads * batch_size
        if phase == 'train':
            image_batch, label_batch = tf.train.shuffle_batch(
                [normalized_image, label], batch_size, capacity,
                min_buffer_size, num_threads)
        else:
            image_batch, label_batch = tf.train.batch(
                [normalized_image, label],
                batch_size, num_threads,
                capacity)
        if data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, [0, 3, 1, 2])
        return image_batch, label_batch


def parse_cifar10_example(filename_queue):
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


def process_image(image, image_height, image_width, distort_image):
    """Applies processing steps to the raw input image.

    Args:
        image: 3D uint8 Tensor representing image in RGB format.
        image_height: int. Height of processed image.
        image_width: width. Width of processed image.
        distort_image: bool. Whether random distortion steps should be applied.
    Returns:
        processed_image: float32 image Tensor [image_height, image_width, 3].
    """
    # TODO: Consider making other resizing options available instead
    with tf.name_scope('image_processing'):
        processed_image = tf.image.convert_image_dtype(image, tf.float32)
        if distort_image:
            with tf.name_scope('image_distortion'):
                processed_image = tf.random_crop(
                    processed_image, [image_height, image_width, 3])
                processed_image = distort_color(processed_image)
        else:
            processed_image = tf.image.resize_image_with_crop_or_pad(
                processed_image, image_height, image_width)
        return processed_image


def distort_color(image):
    """Performs random color distortions in a random order."""
    distort_funcs = [lambda img: tf.image.random_brightness(img, 32.0 / 255),
                     lambda img: tf.image.random_contrast(img, 0.5, 1.5),
                     lambda img: tf.image.random_hue(img, 0.2),
                     lambda img: tf.image.random_saturation(img, 0.5, 1.5)]
    random.shuffle(distort_funcs)
    return functools.reduce(lambda res, f: f(res), distort_funcs, image)


def _get_dims(image):
    with tf.Session() as sess:
        image_val = sess.run(image)
        return len(image_val), len(image_val[0])
