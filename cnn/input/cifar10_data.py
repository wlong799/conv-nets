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
        self._count = 0

    @property
    def image_shape(self):
        return 24, 24, 3

    @property
    def num_classes(self):
        return 10

    def examples_per_epoch(self, phase):
        self._assert_phase(phase)
        if phase == 'train':
            return 40000
        elif phase == 'valid':
            return 10000
        elif phase == 'test':
            return 10000

    def _create_dataset(self, overwrite):
        data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
        cifar_data_dir = 'cifar-10-batches-bin'
        metadata_name = os.path.join(self.data_dir, cifar_data_dir,
                                     'batches.meta.txt')
        data_names = [[
            os.path.join(self.data_dir, cifar_data_dir, filename) for
            filename in ['data_batch_{}.bin'.format(i) for i in range(1, 5)]],
            [os.path.join(self.data_dir, cifar_data_dir, 'data_batch_5.bin')],
            [os.path.join(self.data_dir, cifar_data_dir, 'test_batch.bin')]]
        record_names = [
            os.path.join(self.data_dir, 'cifar10_{}.tfrecords'.format(phase))
            for phase in ['train', 'valid', 'test']]

        # Remove current records if overwrite is on
        if overwrite:
            [os.remove(record_name) for record_name in record_names if
             os.path.exists(record_name)]
        # If all record files already exist, exit
        if all([os.path.exists(record_name) for record_name in record_names]):
            return

        # Download and extract binaries
        tar_filename = os.path.join(self.data_dir, data_url.split('/')[-1])
        if not os.path.exists(tar_filename):
            utils.download_dataset([data_url], self.data_dir)
        with tarfile.open(tar_filename, 'r:gz') as tar_file:
            tar_file.extractall(self.data_dir)

        # Load metadata (i.e. class names) as list
        metadata = _load_cifar10_metadata_from_file(metadata_name)

        # Create TFRecords files that don't already exist
        for record_name, data_name in zip(record_names, data_names):
            if not os.path.exists(record_name):
                _create_cifar10_record_from_binaries(record_name, data_name,
                                                     metadata)

        # Clean up binaries
        shutil.rmtree(os.path.join(self.data_dir, cifar_data_dir))


def _load_cifar10_metadata_from_file(metadata_filename):
    """Loads CIFAR10 class metadata from file and returns a list, such that
    metadata[label] = 'class name'"""
    with open(metadata_filename) as metadata_file:
        metadata = [line for line in metadata_file]
    return metadata


def _create_cifar10_record_from_binaries(record_filename, data_filenames,
                                         metadata):
    """Parses CIFAR-10 binaries and writes examples to TFRecords file.

    The CIFAR-10 binary format is described here:
    http://www.cs.toronto.edu/~kriz/cifar.html

    Args:
        record_filename: Name of TFRecords file to write to.
        data_filenames: List of CIFAR-10 binaries to read from.
        metadata: List such that metadata[i] = 'name of class with label i'
    """
    count = 0
    total_examples = 10000 * len(data_filenames)
    label_bytes = 1
    image_width, image_height, image_channels = 32, 32, 3
    image_bytes = image_width * image_height * image_channels
    coder = utils.ImageCoder()
    example_format = '{}B{}s'.format(label_bytes, image_bytes)

    record_writer = tf.python_io.TFRecordWriter(record_filename)
    for filename in data_filenames:
        with open(filename, 'rb') as binary_reader:
            buffer = binary_reader.read()
        for label, image_string in struct.iter_unpack(example_format, buffer):
            # Reformat byte data into image and encode as JPEG string
            image_1d = np.fromstring(image_string, np.uint8)
            image_3d = image_1d.reshape(
                (image_channels, image_height, image_width))
            image_3d = image_3d.transpose([1, 2, 0])
            encoded_image = coder.encode_jpeg(image_3d)

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': utils.int64_feature(image_height),
                'image/width': utils.int64_feature(image_width),
                'image/channels': utils.int64_feature(image_channels),
                'image/colorspace': utils.bytes_feature('RGB'),
                'image/format': utils.bytes_feature('JPEG'),
                'image/filename': utils.bytes_feature(filename),
                'image/encoded': utils.bytes_feature(encoded_image),
                'class/label': utils.int64_feature(label),
                'class/text': utils.bytes_feature(metadata[label]),
                'bbox/xmin': utils.float_feature(0.0),
                'bbox/xmax': utils.float_feature(1.0),
                'bbox/ymin': utils.float_feature(0.0),
                'bbox/ymax': utils.float_feature(1.0)
            }))
            record_writer.write(example.SerializeToString())
            count += 1
            if count % 100 == 0:
                percent_done = count / total_examples * 100
                print('\r>> Creating dataset {} ({} examples): '
                      '{:>4.1f}% complete'.format(
                    os.path.basename(record_filename), total_examples,
                    percent_done), end='')
    print()
    record_writer.close()
