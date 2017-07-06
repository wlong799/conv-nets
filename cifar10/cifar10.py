# coding=utf-8
"""Program for running analysis on CIFAR-10 dataset"""
import sys

import cnn

if __name__ == '__main__':
    config_section = sys.argv[1]
    phase = sys.argv[2]
    cnn.cnn_app.run('cifar10.ini', config_section, phase=phase)
