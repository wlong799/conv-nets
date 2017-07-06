# coding=utf-8
"""Abstract superclass for datasets."""
import abc
import glob
import os

import tensorflow as tf


class Dataset(metaclass=abc.ABCMeta):
    """Abstract superclass for all datasets."""

    def __init__(self, name, data_dir, overwrite):
        """Sets name and data directory, and creates the dataset."""
        self.name = name
        self.data_dir = data_dir
        self._create_dataset(overwrite)

    @staticmethod
    def _assert_phase(phase):
        """Check that phase is valid and raise ValueError otherwise."""
        if phase not in ['train', 'valid', 'test']:
            raise ValueError('Invalid phase specified: {}'.format(phase))

    def get_filename_queue(self, phase):
        """Returns a queue of filenames for the subset of the dataset
        corresponding to the specified phase."""
        self._assert_phase(phase)
        pattern = '*{}*.tfrecords'.format(phase)
        pattern = os.path.join(self.data_dir, pattern)
        return tf.train.string_input_producer(glob.glob(pattern))

    @abc.abstractmethod
    def _create_dataset(self, overwrite):
        """Creates TFRecords files for the dataset in the data directory.

        Each file must represent a different phase, and have a name matching
        the pattern *{phase}*.tfrecords. See the get_filename_queue() method
        for how the dataset determines the proper data subsets for the
        current phase.

        Only creates the new TFRecords files if they don't already exist.
        However, if self.overwrite is True, any current TFRecords already
        created will be deleted, and new ones must be created.

        Each Example in the dataset should contain the following fields:

        image/height: int64 Image height.
        image/width: int64. Image width.
        image/channels: int64. Image channels.
        image/colorspace: bytes. 'RGB' or 'grayscale'
        image/format: bytes. 'JPEG'
        image/filename: bytes. Filename example was created from.
        image/encoded: bytes. <JPEG encoded string>
        class/label: int64. Label for class of example.
        class/text: bytes. Text name for class of example.
        bbox/xmin: int64. List of xmin values for bounding boxes in image.
        bbox/xmax: int64. List of xmax values for bounding boxes in image.
        bbox/ymin: int64. List of ymin values for bounding boxes in image.
        bbox/ymax: int64. List of ymax values for bounding boxes in image.
        """
        pass

    @property
    @abc.abstractmethod
    def image_shape(self):
        """Get shape image should be post-processing (H, W, C)."""
        pass

    @property
    @abc.abstractmethod
    def num_classes(self):
        """Get number of unique classes/labels contained in dataset."""
        pass

    @abc.abstractmethod
    def examples_per_epoch(self, phase):
        """Number of examples in subset of dataset for the given phase."""
        pass
