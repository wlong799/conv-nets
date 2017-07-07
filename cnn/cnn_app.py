# coding=utf-8
"""Sets configuration and runs the model."""
from copy import deepcopy

import cnn
from multiprocessing import Process


def run(config_file=None, config_section=None, **kwargs):
    """Creates a new ModelConfig using the specified configuration file and
    keyword parameters, and runs the appropriate model."""
    model_config = cnn.config.ModelConfig(config_file, config_section,
                                          **kwargs)
    if model_config.phase == 'test':
        cnn.testing.evaluate(model_config)
    elif model_config.phase == 'valid':
        model_config.valid_repeat_secs = 0
        cnn.testing.evaluate(model_config)
    elif model_config.valid_repeat_secs == 0:
        cnn.training.train(model_config)
    else:
        train_config = model_config
        valid_config = deepcopy(train_config)
        valid_config.phase = 'valid'
        Process(target=cnn.training.train, args=(train_config,)).start()
        Process(target=cnn.testing.evaluate, args=(valid_config,)).start()
