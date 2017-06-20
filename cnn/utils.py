# coding=utf-8
"""Various utility functions (e.g. for downloading and creating datasets)."""
import os

import sys
import tarfile
import urllib.request

import struct
import tensorflow as tf


def download_dataset(download_urls, dest_dir, verbose=True):
    """Downloads the files at the specified urls to the destination directory.

    Args:
        download_urls: list. URLs to download from.
        dest_dir: Destination directory to download files to.
        verbose: bool. Whether or not to show download progress.
    """
    # TODO: Make function Python 2 compatible
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for i, url in enumerate(download_urls):
        filename = url.split('/')[-1]
        filepath = os.path.join(dest_dir, filename)
        # Download file only if it hasn't been already
        if not os.path.exists(filepath):
            def _download_progress(count, block_size, total_size):
                percent_complete = count * block_size / total_size * 100.0
                prog = '\r>> Downloading file {} ({} of {})... {:>4.1f}% ' \
                       'complete'.format(filename, i + 1, len(download_urls),
                                         percent_complete)
                sys.stderr.write(prog)

            if not verbose:
                _download_progress = None
            filepath, _ = urllib.request.urlretrieve(url, filepath,
                                                     _download_progress)
            stat_info = os.stat(filepath)
            if verbose:
                print('Downloaded file {} ({} of {})... {} bytes.'.format(
                    filename, i + 1, len(download_urls),
                    stat_info.st_size))
        else:
            if verbose:
                print(
                    'Skipped file {} ({} of {})... already downloaded'.format(
                        filename, i + 1, len(download_urls)))


def create_cifar10_datasets(dest_dir):
    """Creates a TFRecords file for both the CIFAR-10 train and test data."""
    data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    train_data_names = [os.path.join(dest_dir, filename) for filename in
                        ['data_batch_{}.bin'.format(i) for i in range(1, 6)]]
    test_data_names = [os.path.join(dest_dir, 'test_batch.bin')]
    train_record_name = os.path.join(dest_dir, 'cifar10_train.tfrecords')
    test_record_name = os.path.join(dest_dir, 'cifar10_test.tfrecords')

    # Download file if necessary
    tar_filename = os.path.join(dest_dir, data_url.split('/')[-1])
    if not os.path.exists(tar_filename):
        download_dataset([data_url], dest_dir)
    with tarfile.open(tar_filename, 'r:gz') as tar_file:
        tar_file.extractall(dest_dir)

    # Create TFRecords files
    create_cifar10_record_from_binaries(train_record_name, train_data_names)
    create_cifar10_record_from_binaries(test_record_name, test_data_names)

    # Clean up binaries
    for filename in [tar_filename] + test_data_names + train_data_names:
        os.remove(filename)


def create_cifar10_record_from_binaries(record_filename, data_filenames):
    """Parses the specified binaries and writes the contained examples into
    a single TFRecords file with the given filename.

    The CIFAR-10 binary format is described here:
    http://www.cs.toronto.edu/~kriz/cifar.html
    """
    label_bytes = 1
    image_width, image_height, image_channels = 32, 32, 3
    image_bytes = image_width * image_height * image_channels
    example_format = '{}B{}s'.format(label_bytes, image_bytes)

    record_writer = tf.python_io.TFRecordWriter(record_filename)
    for filename in data_filenames:
        with open(filename, 'rb') as binary_reader:
            buffer = binary_reader.read()
        for label, raw_image in struct.iter_unpack(example_format, buffer):
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[image_height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[image_width])),
                'channels': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[image_channels])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[label])),
                'raw_image': tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[raw_image]))
            }))
            record_writer.write(example.SerializeToString())
    record_writer.close()


if __name__ == '__main__':
    create_cifar10_datasets('data/cifar10')
