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
    appropriate values. """

    def __init__(self, custom_config_file=None,
                 custom_config_section=None, **kwargs):
        """Initializes configuration.

        Configuration parameters are loaded first from the default file,
        then from user specified file (if available), and then from any
        passed keyword arguments. Look at defaults.ini for an example
        configuration file setup, and to see documentation on the
        configuration parameters available.

        Most variables have reasonable default fallback values provided by
        the defaults.ini configuration file. However, certain variables,
        specific to the dataset the model is training on, do not, and thus
        must be specified by the user. See defaults.ini for details.

        Args:
            custom_config_file: Filename of custom configuration file,
                                if available.
            custom_config_section: Name of section in file to load parameters
                                   from. Defaults to DEFAULT_CONFIG_SECTION.
            **kwargs: Configuration parameters passed as keyword arguments.
                      Arguments must be strings, to be compatible with
                      settings from configuration files.
        """
        # Load all available parameters to an initial dictionary
        self._config_dict = {}
        self._load_config(DEFAULT_CONFIG_FILE, DEFAULT_CONFIG_SECTION)
        custom_config_section = custom_config_section or DEFAULT_CONFIG_SECTION
        if custom_config_file:
            self._load_config(custom_config_file, custom_config_section)
        # Initial dictionary expects key and value pairs both to be strings
        for key in kwargs:
            val = kwargs[key]
            if not isinstance(val, str):
                raise ValueError("Keyword arguments must be passed as strings."
                                 " Error with key '{}'".format(key))
            self._config_dict[key] = val

        # Check parameter validity and convert to appropriate type
        self.dataset_name = self._get_string('dataset_name')
        self.overwrite = self._get_bool('overwrite')
        self.phase = self._get_string('phase', ['train', 'valid', 'test'])
        self.model_type = self._get_string('model_type')
        self.padding_mode = self._get_string('padding_mode', ['SAME', 'VALID'])

        self.data_dir = self._get_string('data_dir')
        self.checkpoints_dir = self._get_string('checkpoints_dir')
        self.summaries_dir = self._get_string('summaries_dir')

        self.batch_size = self._get_num('batch_size', int, 1)
        self.distort_images = self._get_bool('distort_images')
        self.num_preprocessing_threads = self._get_num(
            'num_preprocessing_threads', int, 1)
        self.min_example_fraction = self._get_num(
            'min_example_fraction', float, 0, 1)
        self.data_format = self._get_string('data_format', ['NHWC', 'NCHW'])
        data_type_str = self._get_string(
            'data_type', ['float16', 'float32', 'float64'])
        self.data_type = data_type_str == 'float16' and tf.float16 or \
                         data_type_str == 'float32' and tf.float32 or \
                         data_type_str == 'float64' and tf.float64

        self.num_gpus = self._get_num('num_gpus', int, 0)

        self.init_learning_rate = self._get_num('init_learning_rate', float, 0)
        self.learning_decay_rate = self._get_num('learning_decay_rate',
                                                 float, 0, 1)
        self.epochs_per_decay = self._get_num('epochs_per_decay', int, 1)
        self.weight_decay_rate = self._get_num('weight_decay_rate',
                                               float, 0)
        self.moving_avg_decay_rate = self._get_num('moving_avg_decay_rate',
                                                   float, 0, 1)

        # TODO: Fix following line to provide better error checking
        self.top_k_tests = [int(num) for num in
                            self._get_string('top_k_tests').split(',')]
        self.restore_moving_averages = self._get_bool(
            'restore_moving_averages')
        self.valid_set_fraction = self._get_num(
            'valid_set_fraction', float, 0, 1)
        self.valid_repeat_secs = self._get_num('valid_repeat_secs', int, 0)

        self.log_device_placement = self._get_bool('log_device_placement')
        self.print_log_steps = self._get_num('print_log_steps', int, 0)
        self.save_checkpoint_secs = self._get_num(
            'save_checkpoint_secs', int, 0)
        self.save_summaries_steps = self._get_num(
            'save_summaries_steps', int, 0)

    def _load_config(self, config_file, config_section):
        """Loads settings from configuration file."""
        config = configparser.ConfigParser(allow_no_value=True,
                                           inline_comment_prefixes='#')
        config.read(config_file)
        if not config.has_section(config_section):
            raise ValueError("Section {} not found in file {}."
                             .format(config_section, config_file))
        self._config_dict.update(config[config_section])

    def _get_bool(self, key):
        """Converts configuration setting to a boolean value. Expects one of
        'true', '1', 'yes' or 'on' for True; Expects 'false', '0', 'no' or
        'off' for False (capitalization does not matter)."""
        if key not in self._config_dict or self._config_dict[key] is None:
            raise ValueError("Setting '{}' was not specified".format(key))
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
        if key not in self._config_dict or self._config_dict[key] is None:
            raise ValueError("Setting '{}' was not specified".format(key))
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
        if key not in self._config_dict or self._config_dict[key] is None:
            raise ValueError("Setting '{}' was not specified".format(key))
        if choices and self._config_dict[key] not in choices:
            raise ValueError("Setting '{}' must be one of following: '{}'".
                             format(key, ', '.join(choices)))
        return self._config_dict[key]
