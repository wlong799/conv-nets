# coding=utf-8
"""Sets configuration and runs the model."""
import cnn


def run(config_file=None, config_section=None, **kwargs):
    """Creates a new ModelConfig using the specified configuration file and
    keyword parameters, and runs the appropriate model. """
    model_config = cnn.config.ModelConfig(config_file, config_section,
                                          **kwargs)
    if model_config.phase == 'train':
        cnn.train.train(model_config)
    else:
        cnn.eval.evaluate(model_config)
