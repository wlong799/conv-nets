# coding=utf-8
"""Program for running analysis on CIFAR-10 dataset"""
import os
import shutil
import struct
import sys
import tarfile

import numpy as np
import tensorflow as tf

import cnn





if __name__ == '__main__':
    config_section = sys.argv[1]
    phase = sys.argv[2]
    examples_per_epoch = '50000' if phase == 'train' else '10000'
    create_cifar10_datasets('data/')
    cnn.cnn_app.run('cifar10.ini', config_section,
                    phase=phase, examples_per_epoch=examples_per_epoch)
