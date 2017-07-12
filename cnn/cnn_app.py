# coding=utf-8
"""Runs model under the appropriate settings."""
from copy import deepcopy
from multiprocessing import Process

import tensorflow as tf
import time

import cnn


def run(config_file=None, config_section=None, **kwargs):
    """Creates a new ModelConfig using the specified configuration file and
    keyword parameters, and runs the appropriate model."""
    model_config = cnn.config.ModelConfig(config_file, config_section,
                                          **kwargs)
    # Run a single evaluation on test dataset
    if model_config.phase == 'test':
        cnn.testing.evaluate(model_config)
    # Run an evaluation on validation dataset
    # Since not running in background, only run once, on entire dataset
    if model_config.phase == 'valid':
        model_config.bg_valid_set_fraction = 1.0
        model_config.bg_valid_repeat_secs = 0
        cnn.testing.evaluate(model_config)
    # If background validation turned off, just run training session
    elif model_config.bg_valid_repeat_secs == 0:
        cnn.training.train(model_config)
    # Run training and validation in parallel
    else:
        train_config = model_config
        valid_config = deepcopy(train_config)
        valid_config.phase = 'valid'
        Process(target=cnn.training.train, args=(train_config,)).start()
        # Wait for first checkpoint to begin validation
        while tf.train.latest_checkpoint(
                checkpoint_dir=model_config.checkpoints_dir) is None:
            time.sleep(10)
        Process(target=cnn.testing.evaluate, args=(valid_config,)).start()
