# coding=utf-8
"""Provides parameter class specifying model configuration."""
import configparser
import os

import tensorflow as tf

DEFAULT_CONFIG_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'defaults.ini'))
DEFAULT_CONFIG_SECTION = 'model_config'


class ModelConfig(object):
    """Class contains parameters used for configuring a CNN model.

    Parameters can be loaded from configuration files, or from passed
    keyword arguments. Automatically checks that specified parameters have
    appropriate values."""

    def __init__(self, custom_config_file=None,
                 custom_config_section=None, **kwargs):
        """Initializes configuration.

        Configuration parameters are loaded first from the default file,
        then from user specified file (if available), and then from any
        passed keyword arguments. Look at defaults.ini for an example
        configuration file setup, to see documentation on the configuration
        parameters available, and to see the default values for each parameter.

        Args:
            custom_config_file: Filename of custom configuration file,
                                if available.
            custom_config_section: Name of section in file to load parameters
                                   from. Defaults to DEFAULT_CONFIG_SECTION.
            **kwargs: Configuration parameters passed as keyword arguments.
                      Arguments must be strings, to be compatible with
                      settings from configuration files.
        """
        self._config_dict = {}
        self._load_config_to_dict(DEFAULT_CONFIG_FILE, DEFAULT_CONFIG_SECTION)
        if custom_config_file:
            custom_config_section = (custom_config_section or
                                     DEFAULT_CONFIG_SECTION)
            self._load_config_to_dict(custom_config_file,
                                      custom_config_section)

        for config_key, config_val in kwargs.items():
            if not isinstance(config_val, str):
                raise ValueError("Keyword arguments must be passed as strings."
                                 " Error with key '{}'".format(config_key))
            self._config_dict[config_key] = config_val

        self.dataset_name = self._get_string('dataset_name')
        try:
            self.dataset_config = self._get_string('dataset_config')
        except ValueError:
            self.dataset_config = None
        self.overwrite = self._get_bool('overwrite')

        self.model_name = self._get_string('model_name')
        self.phase = self._get_string('phase', ['train', 'valid', 'test'])
        self.use_batch_norm = self._get_bool('use_batch_norm')
        self.padding_mode = self._get_string('padding_mode', ['same', 'valid'])

        self.data_dir = self._get_string('data_dir')
        self.checkpoints_dir = self._get_string('checkpoints_dir')
        self.summaries_dir = self._get_string('summaries_dir')

        self.batch_size = self._get_num('batch_size', int, 1)
        self.distort_images = self._get_bool('distort_images')
        self.num_preprocessing_threads = self._get_num(
            'num_preprocessing_threads', int, 1)
        self.num_readers = self._get_num('num_readers', int, 1)
        self.min_example_fraction = self._get_num(
            'min_example_fraction', float, 0, 1)

        self.num_gpus = self._get_num('num_gpus', int, 0)
        self.data_format = 'NCHW' if self.num_gpus else 'NHWC'

        self.init_learning_rate = self._get_num('init_learning_rate', float, 0)
        self.learning_decay_rate = self._get_num('learning_decay_rate',
                                                 float, 0, 1)
        self.epochs_per_decay = self._get_num('epochs_per_decay', int, 1)
        self.momentum = self._get_num('momentum', float, 0, 1)
        self.weight_decay_rate = self._get_num('weight_decay_rate', float, 0)
        self.moving_avg_decay_rate = self._get_num('moving_avg_decay_rate',
                                                   float, 0, 1)

        # Loading of top_k_tests is a bit hacky to ensure error checking
        top_k_tests_str = self._get_string('top_k_tests').split(',')
        top_k_tests = []
        for str_val in top_k_tests_str:
            self._config_dict['top_k_tests_split'] = str_val
            top_k_tests.append(self._get_num('top_k_tests_split', int, 1))
        self.top_k_tests = top_k_tests
        self.bg_valid_set_fraction = self._get_num(
            'bg_valid_set_fraction', float, 0, 1)
        self.bg_valid_repeat_secs = self._get_num(
            'bg_valid_repeat_secs', int, 0)

        self.log_device_placement = self._get_bool('log_device_placement')
        self.print_log_steps = self._get_num('print_log_steps', int, 0)
        self.save_checkpoint_secs = self._get_num(
            'save_checkpoint_secs', int, 0)
        self.save_summaries_steps = self._get_num(
            'save_summaries_steps', int, 0)

    def _load_config_to_dict(self, config_file, config_section):
        """Loads settings from configuration file."""
        config = configparser.ConfigParser(allow_no_value=True,
                                           inline_comment_prefixes='#')
        config.read(config_file)
        if not config.has_section(config_section):
            raise ValueError("Section {} not found in file {}."
                             .format(config_section, config_file))
        self._config_dict.update(config[config_section])

    def _assert_key(self, key):
        """Check that key and associated value actually exist."""
        if key not in self._config_dict or self._config_dict[key] is None:
            raise ValueError("Setting '{}' was not specified".format(key))

    def _get_bool(self, key):
        """Converts configuration setting to a boolean value. Expects one of
        'true', '1', 'yes' or 'on' for True; Expects 'false', '0', 'no' or
        'off' for False (capitalization does not matter)."""
        self._assert_key(key)
        true_vals = ['true', '1', 'yes', 'on']
        false_vals = ['false', '0', 'no', 'off']
        if self._config_dict[key].lower() in true_vals:
            return True
        elif self._config_dict[key].lower() in false_vals:
            return False
        else:
            raise ValueError("Setting '{}' must be a boolean".format(key))

    def _get_num(self, key, num_type, min_val=None, max_val=None):
        """Converts configuration setting to numeric value of specified
        type, and ensures that it falls within appropriate bounds,
        if provided."""
        self._assert_key(key)
        try:
            val = num_type(self._config_dict[key])
        except ValueError as e:
            e.args = e.args or ('',)
            e.args += ("Setting '{}' requires type of {}".format(
                key, num_type),)
            raise

        if min_val and val < min_val:
            raise ValueError("Setting '{}' out of range: {} < {}.".format(
                key, val, min_val))
        if max_val and val > max_val:
            raise ValueError("Setting '{}' out of range: {} > {}.".format(
                key, val, max_val))

        return val

    def _get_string(self, key, choices=None):
        """Returns configuration setting as string, and checks that it is
        one of the allowed choices, if provided. """
        self._assert_key(key)
        if choices and self._config_dict[key] not in choices:
            raise ValueError("Setting '{}' must be one of following: '{}'".
                             format(key, ', '.join(choices)))
        return self._config_dict[key]
