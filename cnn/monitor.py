# coding=utf-8
"""Custom session monitor for CNNs"""
import datetime
import time

import tensorflow as tf

import cnn


def create_training_session(model_config: cnn.config.ModelConfig,
                            loss, global_step):
    """Creates a MonitoredSession for training the model.

    This monitored session automatically saves and restores from checkpoints,
    saves variable summaries for visualization in TensorBoard, and logs
    basic info to the terminal. This is useful for monitoring models during
    long training sessions, and ensuring that they can be saved in case of
    program failure. Should only be used during train phase.

    Args:
        model_config: Model configuration.
        loss: Loss tensor to track for logging.
        global_step: Global step tensor used for logging.

    Returns:
        MonitoredSession for training the model.
    """
    if model_config.phase != 'train':
        raise ValueError(
            "Only use create_training_session for training phase.")
    # Allow soft placement for operations to be placed on proper device
    # Log device placement allows for debugging of device use if desired
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=model_config.log_device_placement)
    # Session creator restores from checkpoint with option for custom restores
    scaffold = tf.train.Scaffold()
    session_creator = tf.train.ChiefSessionCreator(
        scaffold, config=config, checkpoint_dir=model_config.checkpoints_dir)

    hooks = []
    # Saves checkpoints after certain number of seconds
    if model_config.save_checkpoint_secs > 0:
        hooks.append(tf.train.CheckpointSaverHook(
            model_config.checkpoints_dir, model_config.save_checkpoint_secs,
            scaffold=scaffold))
    if model_config.save_summaries_steps > 0:
        # Saves number of model steps executed per second
        hooks.append(tf.train.StepCounterHook(
            model_config.save_summaries_steps,
            output_dir=model_config.summaries_dir))
        # Saves all variable summaries after certain number of steps
        hooks.append(tf.train.SummarySaverHook(
            model_config.save_summaries_steps,
            output_dir=model_config.summaries_dir, scaffold=scaffold))
    # Logs loss and step information to terminal
    hooks.append(tf.train.NanTensorHook(loss))
    if model_config.print_log_steps > 0:
        examples_per_step = (model_config.batch_size *
                             (model_config.num_gpus or 1))
        hooks.append(
            _LoggerHook(loss, global_step, examples_per_step,
                        model_config.print_log_steps))

    return tf.train.MonitoredSession(session_creator, hooks)


def create_testing_session(model_config: cnn.config.ModelConfig, saver):
    """Creates a MonitoredSession for testing the model.

    This monitored session allows a Saver argument for custom restores (e.g.
    from moving variable averages), but contains no hooks for saving
    summaries/checkpoints to avoid conflicts with training summaries and
    prevent training on the testing/validation sets. Should be used for
    validation and testing phases.

    Args:
        model_config: Model configuration.
        saver: Saver to use for restoring variables.

    Returns:
        MonitoredSession for testing the model.
    """
    if model_config.phase not in ['valid', 'test']:
        raise ValueError(
            "Only use create_testing_session for validation or testing phase.")

    scaffold = tf.train.Scaffold(saver=saver)
    session_creator = tf.train.ChiefSessionCreator(
        scaffold, checkpoint_dir=model_config.checkpoints_dir)

    return tf.train.MonitoredSession(session_creator)


# noinspection PyMissingOrEmptyDocstring
class _LoggerHook(tf.train.SessionRunHook):
    """Logs run speed and value of loss tensor to terminal."""

    def __init__(self, loss, global_step, examples_per_step, log_frequency):
        self.loss = loss
        self.global_step = global_step
        self.examples_per_step = examples_per_step
        self.log_frequency = log_frequency
        self.start_time = time.time()

    def begin(self):
        self.start_time = time.time()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.loss, self.global_step])

    def after_run(self, run_context, run_values):
        loss_value, global_step_value = run_values.results
        if global_step_value % self.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self.start_time
            self.start_time = current_time

            secs_per_step = float(duration / self.log_frequency)
            examples_per_sec = self.examples_per_step / secs_per_step
            format_str = '{}: step {} | Loss = {:.2f} | ' \
                         '{:.1f} examples/second | {:.3f} seconds/step'
            print(format_str.format(datetime.datetime.now(), global_step_value,
                                    loss_value, examples_per_sec,
                                    secs_per_step))
