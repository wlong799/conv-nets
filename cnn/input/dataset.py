# coding=utf-8
"""Abstract superclass for datasets."""
import abc
import glob
import os

import tensorflow as tf


class Dataset(metaclass=abc.ABCMeta):
    """Abstract superclass for all datasets. Implementations must be able to
    build the initial dataset into TFRecords files, and provide basic
    information about the dataset. """

    def __init__(self, name, data_dir, overwrite):
        """Sets name and data directory, as well as whether to force
        creation of a new dataset, even if dataset currently exists. """
        self.name = name
        self.data_dir = data_dir
        self.overwrite = overwrite

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

    @property
    @abc.abstractmethod
    def image_shape(self):
        """Get the shape that image should be post-processing (H, W, C)."""
        pass

    @property
    @abc.abstractmethod
    def num_classes(self):
        """Get tbe number of unique classes/labels contained in dataset."""
        pass

    def examples_per_epoch(self, phase):
        """Get number of examples in the subset of data for the given phase."""
        self._assert_phase(phase)
        return self._examples_per_epoch(phase)

    @abc.abstractmethod
    def _examples_per_epoch(self, phase):
        """Helper for examples_per_epoch() to be overwritten. Phase
        guaranteed to be one of 'train', 'valid', or 'test'. """
        pass

    def create_dataset(self):
        """Creates TFRecords files for the dataset in the data directory.

        Data for different phases must be located in different files,
        and have a name matching the pattern *{phase}*.tfrecords. See the
        get_filename_queue() method for how the dataset determines the
        proper data files for the current phase.

        Only creates the new TFRecords files if they don't already exist.
        However, if self.overwrite is True, any current TFRecords already
        created will be deleted, and new ones must be created.

        Each serialized example in the records files should contain the
        following fields:

        image/encoded: bytes. <JPEG encoded string>.
        class/label: int64. Label for class of example.
        class/text: bytes. Text label for class of example.
        """
        # Remove files if overwrite is on
        if self.overwrite:
            patterns = [os.path.join(
                self.data_dir, '*{}*.tfrecords'.format(phase)) for phase in
                ['train', 'valid', 'test']]
            for pattern in patterns:
                filenames = glob.glob(pattern)
                [os.remove(filename) for filename in filenames]
        # Create necessary datasets and check everything has been initialized
        if not self._all_records_files_created:
            self._create_dataset()
        if not self._all_records_files_created:
            raise RuntimeError(
                "Not all files were initialized for dataset '{}'.".format(
                    self.name))

    @property
    @abc.abstractmethod
    def _all_records_files_created(self):
        """Should return True only if all expected TFRecords files have been
        created."""
        pass

    @abc.abstractmethod
    def _create_dataset(self):
        """Only runs if at least some expected TFRecords files are not
        present, and at end, all expected files must have been created. """
        pass
