# coding=utf-8
"""Custom session monitor for CNNs"""
import datetime
import time

import tensorflow as tf

import cnn


def get_monitored_cnn_session(model_config: cnn.config.ModelConfig,
                              loss=None, global_step=None, saver=None):
    """Creates a MonitoredSession for running the model.

    The monitored session automatically saves and restores from checkpoints,
    saves variable summaries for visualization in TensorBoard, and logs
    basic info to the terminal. This is useful for monitoring models during
    long training sessions, and ensuring that they can be saved in case of
    program failure.

    Args:
        model_config: Model configuration.
        loss: Loss tensor to track for logging. Both it and the global step
              tensor must be provided for logging to occur.
        global_step: Global step tensor used for logging.
        saver: Saver for custom restore of variables (e.g. for reloading using
               moving average rather than raw variable).

    Returns:
        MonitoredSession for running the model.
    """
    # Allow soft placement for operations to be placed on proper device
    # Turn log device placement on to debug device use
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=model_config.log_device_placement)
    # Session creator restores from checkpoint with option for custom restores
    scaffold = tf.train.Scaffold(saver=saver)
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
    if loss is not None and global_step is not None:
        hooks.append(tf.train.NanTensorHook(loss))
        if model_config.batch_size > 0 and model_config.print_log_steps > 0:
            hooks.append(
                _LoggerHook(loss, global_step, model_config.batch_size,
                            model_config.print_log_steps))

    # Only use hooks for training sessions
    if model_config.phase == 'train':
        return tf.train.MonitoredSession(session_creator, hooks)
    else:
        return tf.train.MonitoredSession(session_creator)


# noinspection PyMissingOrEmptyDocstring
class _LoggerHook(tf.train.SessionRunHook):
    """Logs run speed and value of loss tensor to terminal."""

    def __init__(self, loss, global_step, batch_size, log_frequency):
        self.loss = loss
        self.global_step = global_step
        self.batch_size = batch_size
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

            examples_per_sec = self.log_frequency * self.batch_size / duration
            secs_per_batch = float(duration / self.log_frequency)
            format_str = '{}: step {} | Loss = {:.2f} | ' \
                         '{:.1f} examples/second | {:.3f} seconds/batch)'
            print(format_str.format(datetime.datetime.now(), global_step_value,
                                    loss_value, examples_per_sec,
                                    secs_per_batch))
