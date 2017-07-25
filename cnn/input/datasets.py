# coding=utf-8
"""Module contains abstract superclasses for datasets.

Dataset classes provide information about the specific data they contain,
so they can be properly parsed/analyzed by the model.

Implementing classes should subclass BasicDataset if they don't require any
runtime configuration (i.e. dataset creation and the information contained
is the same every time). They should subclass ConfigDataset if they require
runtime configuration in the form of a configuration file (e.g. if user
wishes to include/exclude certain examples from dataset in different runs)."""
import abc
import glob
import os
import pickle
from collections import defaultdict

import tensorflow as tf


class Dataset(metaclass=abc.ABCMeta):
    """Abstract superclass for BasicDataset and ConfigDataset.

    Implementation classes should subclass one of those two classes,
    not this one. Provides methods for building initial dataset into TFRecords
    files, and getting basic information about the dataset. Template method
    design pattern used to provide basic structure and error checking for all
    subclasses."""

    def __init__(self, data_dir, overwrite):
        """Sets up initial configuration of dataset.

        Args:
            data_dir: Directory to read/write data.
            overwrite: bool. If True, create_dataset() will recreate all
                       TFRecords files from scratch, even if they already
                       exist.
        """
        self._data_dir = data_dir
        self._overwrite = overwrite
        self._filename_queue = None
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    @staticmethod
    @abc.abstractmethod
    def name():
        """Each Dataset subclass should have its own unique name. This name
        will be used to select the appropriate class according to the
        specified model configuration."""
        pass

    def examples_per_epoch(self, phase):
        """Number of examples in the subset of data for the given phase.

        Subclasses must implement proper behavior through the abstract helper
        _examples_per_epoch()."""
        self._assert_phase(phase)
        return self._examples_per_epoch(phase)

    @abc.abstractmethod
    def _examples_per_epoch(self, phase):
        pass

    @property
    @abc.abstractmethod
    def image_shape(self):
        """Shape that image should be post-processing (H, W, C).

        Note that this does not mean that the raw images stored in the
        dataset are of this size."""
        pass

    @property
    @abc.abstractmethod
    def num_classes(self):
        """Number of unique classes/labels contained in dataset."""
        pass

    @property
    def class_weights(self):
        """Relative importance of each class when calculating loss.

        In some cases, it is desired to weight certain classes more than
        others when calculating model loss. For example, in cases where
        there is class imbalance in the dataset, one should weight each
        class by the reciprocal of their frequency, so that the classes with
        more training examples available do not overshadow the others.

        Subclasses must implement proper behavior through the abstract
        helper _class_weights().

        Returns:
            weights: float list of size self.num_classes. weights[id] is the
                     loss weighting factor that should be used for class with
                     label 'id'.
                     None if all classes should be weighted the same.
        """
        weights = self._class_weights()
        if weights is not None:
            if not isinstance(weights, list):
                raise ValueError("Class weights must be provided as list.")
            if len(weights) != self.num_classes:
                raise ValueError("Class weights must be same length as number "
                                 "of dataset classes.")
        return weights

    @abc.abstractmethod
    def _class_weights(self):
        pass

    def create_dataset(self):
        """Creates the dataset TFRecords and metadata files in self.data_dir.

        Only creates the new files if they don't exist already, unless
        self.overwrite is True, in which case all existing files will be
        recreated as well.

        Subclasses must implement proper behavior of actual dataset creation
        through the abstract helper _create_dataset().

        Each serialized example in the records files should contain the
        following fields:

        image/encoded: bytes. <JPEG encoded string>.
        class/label: int64. Label for class of example.
        class/text: bytes. Text label for class of example.
        """
        with tf.name_scope('{}_dataset_creation'.format(self.name())):
            if self._overwrite:
                [os.remove(filename) for filename in self._get_all_data_files()
                 if os.path.exists(filename)]

            if all([os.path.exists(filename) for filename in
                    self._get_all_data_files()]):
                return
            else:
                self._create_dataset()
                if not all([os.path.exists(filename) for filename in
                            self._get_all_data_files()]):
                    raise RuntimeError("Not all files were initialized for "
                                       "dataset '{}'.".format(self.name()))

    @abc.abstractmethod
    def _create_dataset(self):
        pass

    def read_example(self, phase):
        """Reads and parses next example for the specified phase.

        Args:
            phase: Phase to parse examples from.

        Returns:
            image: 3D uint8 Tensor [height, width, channels]. Raw image in RGB
                   if channels == 3, or grayscale if channels == 1.
            label: int32 Tensor. The input class label.
            text: string Tensor. The input class name.
        """
        self._assert_phase(phase)
        # Obtain proper queue of filenames and reader for specified phase
        with tf.name_scope('file_preparation'):
            filenames = self._get_records(phase)
            self._filename_queue = self._filename_queue or \
                                   tf.train.string_input_producer(filenames)
            reader = tf.TFRecordReader()
        with tf.name_scope('example_parsing'):
            _, serialized_example = reader.read(self._filename_queue)
            features = tf.parse_single_example(serialized_example, features={
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'class/label': tf.FixedLenFeature([], tf.int64),
                'class/text': tf.FixedLenFeature([], tf.string)
            })
            image = tf.image.decode_jpeg(features['image/encoded'])
            label = tf.cast(features['class/label'], tf.int32)
            text = features['class/text']

        return image, label, text

    @abc.abstractmethod
    def _get_all_data_files(self) -> list:
        """Returns all files that belong to the complete created dataset,
        including all TFRecords files for each phase, and any metadata
        files."""
        pass

    def _get_records(self, phase):
        """Returns all record filenames for the specified phase.

        Subclasses must implement proper behavior through the abstract
        helper _get_records_helper()."""
        self._assert_phase(phase)
        return self._get_records_helper(phase)

    @abc.abstractmethod
    def _get_records_helper(self, phase) -> list:
        pass

    @staticmethod
    def _assert_phase(phase):
        if phase not in ['train', 'valid', 'test']:
            raise ValueError('Invalid phase specified: {}'.format(phase))


# noinspection PyAbstractClass
class BasicDataset(Dataset):
    """Abstract superclass for datasets that do not require runtime
    configuration."""

    def _get_all_data_files(self):
        filenames = []
        for phase in ['train', 'valid', 'test']:
            filenames.extend(self._get_records(phase))
        return filenames

    def _get_records_helper(self, phase):
        return [os.path.join(self._data_dir, '{}.tfrecords'.format(phase))]


# noinspection PyMissingOrEmptyDocstring,PyAbstractClass
class ConfigDataset(Dataset):
    """Abstract superclass for datasets that are configured at runtime. Each
    subclass will interpret config files in their own way. Information
    should be saved to the metadata object for retrieval in between runs."""

    def __init__(self, data_dir, overwrite, config_file):
        super().__init__(data_dir, overwrite)
        self._config_file = config_file
        self._metadata_file = os.path.join(data_dir, 'metadata.pkl')
        self._metadata = self._load_metadata()

        if (config_file != self._metadata.config_file or
                    os.path.getmtime(config_file) !=
                    self._metadata.config_last_modified) and \
                not self._overwrite:
            raise RuntimeError("Configuration file has been edited. Overwrite "
                               "must be set to True to recreate datasets.")

    def _examples_per_epoch(self, phase):
        return self._metadata.num_examples_per_phase[phase]

    @property
    def image_shape(self):
        shape = self._metadata.image_shape
        if shape is None:
            raise ValueError("Image shape was not specified in metadata.")
        if not (isinstance(shape, tuple) or isinstance(shape, list)) or \
                        len(shape) != 3:
            raise ValueError("Image shape must be provided as tuple (H, W, C)."
                             " Got {}".format(shape))
        return shape

    def num_classes(self):
        return self._metadata.num_classes

    def _class_weights(self):
        return self._metadata.class_weights

    def _get_all_data_files(self):
        record_filenames = []
        for phase in ['train', 'valid', 'test']:
            record_filenames.extend(self._get_records(phase))
        all_filenames = record_filenames + [self._metadata_file]
        return all_filenames

    def _get_records_helper(self, phase):
        return self._metadata.filenames_per_phase[phase]

    def _load_metadata(self):
        if os.path.exists(self._metadata_file):
            with open(self._metadata_file, 'rb') as pickle_file:
                return pickle.load(pickle_file)
        else:
            return self._Metadata(self._config_file)

    def _save_metadata(self):
        with open(self._metadata_file, 'wb') as pickle_file:
            pickle.dump(self._metadata, pickle_file)

    # noinspection PyMissingOrEmptyDocstring
    class _Metadata(object):
        """Simple container class to store metadata for dataset. Necessary for
        configurable datasets, because their information (e.g. number of
        classes, number of examples) can change depending on the
        configuration file, so must have a way to save and load this
        information in between runs.

        Implementing classes of ConfigDataset do not necessarily need to save
        all the below instance variables in their metadata. But if they do
        not, they will need to override the appropriate method in
        ConfigDataset to provide the information in a different way."""

        def __init__(self, config_file):
            # Saving config file allows us to assert that configuration has
            # not been altered in between runs.
            self.config_file = config_file
            self.config_last_modified = os.path.getmtime(config_file)

            self.num_examples_per_phase = {
                'train': 0,
                'valid': 0,
                'test': 0
            }
            self.image_shape = None
            self.num_classes = 0
            self.class_weights = []
            self.filenames_per_phase = defaultdict(list)
            # Any other miscellaneous data should be stored here
            self.other = {}
