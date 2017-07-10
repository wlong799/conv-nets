# coding=utf-8
"""CIFAR10 Dataset"""
import os
import shutil
import struct
import tarfile

import numpy as np
import tensorflow as tf

from . import utils
from .dataset import Dataset


# noinspection PyMissingOrEmptyDocstring
class CIFAR10Data(Dataset):
    def __init__(self, data_dir, overwrite):
        super(CIFAR10Data, self).__init__('CIFAR10', data_dir, overwrite)
        self._data_url = \
            'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
        self._binaries_dir = os.path.join(self.data_dir,
                                          'cifar-10-batches-bin')

        self._metadata_name = os.path.join(self._binaries_dir,
                                           'batches.meta.txt')

        self._data_filenames_dict = {
            'train': [os.path.join(self._binaries_dir, filename) for filename
                      in ['data_batch_{}.bin'.format(i) for i in range(1, 5)]],
            'valid': [os.path.join(self._binaries_dir, 'data_batch_5.bin')],
            'test': [os.path.join(self._binaries_dir, 'test_batch.bin')]
        }

    @property
    def image_shape(self):
        return 24, 24, 3

    @property
    def num_classes(self):
        return 10

    def _examples_per_epoch(self, phase):
        if phase == 'train':
            return 40000
        elif phase == 'valid':
            return 10000
        elif phase == 'test':
            return 10000

    @property
    def _all_records_files_created(self):
        return all([os.path.exists(self._get_phase_record_name(phase))
                    for phase in ['train', 'valid', 'test']])

    def _create_dataset(self):
        # Download and extract binaries
        tar_filename = os.path.join(self.data_dir,
                                    self._data_url.split('/')[-1])
        if not os.path.exists(tar_filename):
            utils.download_dataset([self._data_url], self.data_dir)
        with tarfile.open(tar_filename, 'r:gz') as tar_file:
            tar_file.extractall(self.data_dir)

        # Create TFRecords files if they don't already exist
        for phase in self._data_filenames_dict:
            if not os.path.exists(self._get_phase_record_name(phase)):
                self._create_cifar10_record_from_binaries(phase)

        # Clean up binaries
        shutil.rmtree(self._binaries_dir)

    def _create_cifar10_record_from_binaries(self, phase):
        """Parses CIFAR-10 binaries and writes examples to TFRecords file.

        The CIFAR-10 binary format is described here:
        http://www.cs.toronto.edu/~kriz/cifar.html

        Args:
            record_filename: Name of TFRecords file to write to.
            data_filenames: List of CIFAR-10 binaries to read from.
        """
        examples_written = 0
        total_examples = self._examples_per_epoch(phase)

        label_bytes = 1
        image_width, image_height, image_channels = 32, 32, 3
        image_bytes = image_width * image_height * image_channels
        coder = utils.ImageCoder()
        example_format = '{}B{}s'.format(label_bytes, image_bytes)

        record_filename = self._get_phase_record_name(phase)
        data_filenames = self._data_filenames_dict[phase]
        metadata = self._load_cifar10_metadata()

        record_writer = tf.python_io.TFRecordWriter(record_filename)
        for filename in data_filenames:
            with open(filename, 'rb') as binary_reader:
                buffer = binary_reader.read()
            for label, image_string in struct.iter_unpack(example_format,
                                                          buffer):
                # Reformat byte data into image and encode as JPEG string
                image_1d = np.fromstring(image_string, np.uint8)
                image_3d = image_1d.reshape(
                    (image_channels, image_height, image_width))
                image_3d = image_3d.transpose([1, 2, 0])
                encoded_image = coder.encode_jpeg(image_3d)

                # Create and write example
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded': utils.bytes_feature(encoded_image),
                    'class/label': utils.int64_feature(label),
                    'class/text': utils.bytes_feature(metadata[label]),
                }))
                record_writer.write(example.SerializeToString())

                # Log progress
                examples_written += 1
                if examples_written % 500 == 0:
                    percent_done = examples_written / total_examples * 100
                    print('\r>> Creating dataset {} ({} examples): '
                          '{:>4.1f}% complete'.format(
                        os.path.basename(record_filename), total_examples,
                        percent_done), end='')
        print()
        record_writer.close()
        if examples_written != total_examples:
            raise RuntimeError(
                "Number of examples written ({}) did not match"
                "expected ({}) for phase '{}'.".format(
                    examples_written, total_examples, phase))

    def _get_phase_record_name(self, phase):
        """Returns name of record associated with specified phase."""
        return os.path.join(self.data_dir,
                            'cifar10_{}.tfrecords'.format(phase))

    def _load_cifar10_metadata(self):
        """Loads CIFAR10 class metadata from file and returns a list, such that
        metadata[label] = 'class name'."""
        with open(self._metadata_name) as metadata_file:
            metadata = [line for line in metadata_file]
        return metadata
