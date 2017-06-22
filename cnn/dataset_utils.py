# coding=utf-8
"""Various utility functions for downloading and creating datasets."""
import os
import sys
import urllib.request

import tensorflow as tf


def int64_feature(value):
    """Wraps value in a TensorFlow Feature to be stored in a TFRecords file."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """Wraps value in a TensorFlow Feature to be stored in a TFRecords file."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class ImageCoder(object):
    """Helper class to provide TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single session for all image coding calls
        self._sess = tf.Session()

        # Converts 3D uint8 Tensor (RGB image) to encoded PNG string
        self._image_data = tf.placeholder(tf.uint8)
        self._encode_png = tf.image.encode_png(self._image_data)

    def encode_png(self, image_data):
        """Runs session to get evaluated PNG string encoding."""
        image = self._sess.run(self._encode_png,
                               feed_dict={self._image_data: image_data})
        return image


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
                print('\r>> Downloaded file {} ({} of {})... {} bytes.'.format(
                    filename, i + 1, len(download_urls),
                    stat_info.st_size))
        else:
            if verbose:
                print('\r>> Skipped file {} ({} of {})... already downloaded'
                      .format(filename, i + 1, len(download_urls)))



