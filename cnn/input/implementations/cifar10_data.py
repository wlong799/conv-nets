# coding=utf-8
"""CIFAR-10 dataset. 60000 labeled examples of tiny images with 10 classes.
Described in detail here: http://www.cs.toronto.edu/~kriz/cifar.html"""
import os
import shutil
import struct
import tarfile

import numpy as np
import tensorflow as tf

from cnn.input import utils
from cnn.input.datasets import BasicDataset

_DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
_BINARIES_DIR = 'cifar-10-batches-bin'
_CLASS_INFO_FILE = 'batches.meta.txt'
_TRAIN_FILES = ['data_batch_{}.bin'.format(i) for i in range(1, 5)]
_VALID_FILES = ['data_batch_5.bin']
_TEST_FILES = ['test_batch.bin']


# noinspection PyMissingOrEmptyDocstring
class CIFAR10Data(BasicDataset):
    def __init__(self, data_dir, overwrite):
        super().__init__(data_dir, overwrite)
        self._binaries_dir = os.path.join(self._data_dir, _BINARIES_DIR)
        self._class_info_filename = os.path.join(self._binaries_dir,
                                                 _CLASS_INFO_FILE)
        self._data_filenames_dict = {
            'train': [os.path.join(self._binaries_dir, filename) for
                      filename in _TRAIN_FILES],
            'valid': [os.path.join(self._binaries_dir, filename) for
                      filename in _VALID_FILES],
            'test': [os.path.join(self._binaries_dir, filename) for
                     filename in _TEST_FILES],
        }

    @staticmethod
    def get_name():
        return 'cifar10'

    def _examples_per_epoch(self, phase):
        if phase == 'train':
            return 40000
        elif phase == 'valid':
            return 10000
        elif phase == 'test':
            return 10000

    @property
    def image_shape(self):
        return 24, 24, 3

    @property
    def num_classes(self):
        return 10

    def _class_weights(self):
        return None

    def _create_dataset(self):
        tar_filename = os.path.join(self._data_dir, _DATA_URL.split('/')[-1])
        if not os.path.exists(tar_filename):
            utils.download_dataset([_DATA_URL], self._data_dir)
        with tarfile.open(tar_filename, 'r:gz') as tar_file:
            tar_file.extractall(self._data_dir)

        for phase in self._data_filenames_dict:
            phase_record = self._get_records(phase)[0]
            if not os.path.exists(phase_record):
                self._create_cifar10_record_from_binaries(phase)

        shutil.rmtree(self._binaries_dir)

    def _create_cifar10_record_from_binaries(self, phase):
        # CIFAR-10 binary format: http://www.cs.toronto.edu/~kriz/cifar.html
        label_bytes = 1
        image_width, image_height, image_channels = 32, 32, 3
        image_bytes = image_width * image_height * image_channels
        example_format = '{}B{}s'.format(label_bytes, image_bytes)

        data_filenames = self._data_filenames_dict[phase]
        record_filename = self._get_records(phase)[0]
        class_info = self._load_cifar10_class_info()
        coder = utils.ImageCoder()
        with tf.python_io.TFRecordWriter(record_filename) as record_writer:
            examples_written = 0
            total_examples = self._examples_per_epoch(phase)
            for filename in data_filenames:
                with open(filename, 'rb') as binary_reader:
                    buffer = binary_reader.read()
                for label, image_string in struct.iter_unpack(example_format,
                                                              buffer):
                    image_1d = np.fromstring(image_string, np.uint8)
                    image_3d = image_1d.reshape(
                        (image_channels, image_height, image_width))
                    image_3d_reshaped = image_3d.transpose([1, 2, 0])
                    encoded_image = coder.encode_jpeg(image_3d_reshaped)
                    name = class_info[label]

                    features = tf.train.Features(feature={
                        'image/encoded': utils.bytes_feature(encoded_image),
                        'class/label': utils.int64_feature(label),
                        'class/text': utils.bytes_feature(name),
                    })
                    example = tf.train.Example(features=features)
                    serialized_example = example.SerializeToString()
                    record_writer.write(serialized_example)

                    examples_written += 1
                    if examples_written % 250 == 0:
                        percent_done = examples_written / total_examples * 100
                        print('\r>> Creating dataset {} ({} examples): '
                              '{:>4.1f}% complete'.format(
                            os.path.basename(record_filename),
                            total_examples, percent_done), end='', flush=True)
        print()
        if examples_written != total_examples:
            raise RuntimeError(
                "Number of examples written ({}) did not match"
                "expected ({}) for phase '{}'.".format(
                    examples_written, total_examples, phase))

    def _load_cifar10_class_info(self):
        """Loads CIFAR10 class metadata from file and returns a list, such that
        metadata[label] = 'class name'."""
        if not os.path.exists(self._class_info_filename):
            return None
        with open(self._class_info_filename) as metadata_file:
            metadata = [line for line in metadata_file]
        return metadata
