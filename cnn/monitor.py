# coding=utf-8
"""Custom session monitor for CNNs"""
import datetime
import time

import tensorflow as tf

import cnn


def get_monitored_cnn_session(loss, model_config: cnn.config.ModelConfig):
    """Creates a MonitoredSession for running the model.

    The monitored session automatically saves and restores from checkpoints,
    saves summaries, and logs basic info to the terminal. This is useful for
    monitoring models during long training sessions, and ensuring that they
    can be saved in case of program failure.

    Args:
        loss: Loss tensor to track for logging. None to disable logging.
        model_config: Model configuration.

    Returns:
        MonitoredSession for running the model.
    """
    # Session creator that can restore from checkpoint
    scaffold = tf.train.Scaffold()
    session_creator = tf.train.ChiefSessionCreator(
        scaffold, checkpoint_dir=model_config.checkpoints_dir)

    # Create hooks
    hooks = []
    if model_config.save_checkpoint_secs > 0:
        hooks.append(tf.train.CheckpointSaverHook(
            model_config.checkpoints_dir, model_config.save_checkpoint_secs,
            scaffold=scaffold))
    if model_config.save_summaries_steps > 0:
        hooks.append(tf.train.StepCounterHook(
            model_config.save_summaries_steps,
            output_dir=model_config.summaries_dir))
        hooks.append(tf.train.SummarySaverHook(
            model_config.save_summaries_steps,
            output_dir=model_config.summaries_dir, scaffold=scaffold))
    if loss is not None:
        hooks.append(tf.train.NanTensorHook(loss))
        if model_config.batch_size > 0 and model_config.print_log_steps > 0:
            hooks.append(_LoggerHook(loss, model_config.batch_size,
                                     model_config.print_log_steps))

    # Only use hooks for training sessions
    if model_config.phase == 'train':
        return tf.train.MonitoredSession(session_creator, hooks)
    else:
        return tf.train.MonitoredSession(session_creator)


# noinspection PyMissingOrEmptyDocstring
class _LoggerHook(tf.train.SessionRunHook):
    """Logs runtime and values of specified tensors"""

    def __init__(self, loss, batch_size, log_frequency):
        self.loss = loss,
        self.batch_size = batch_size
        self.log_frequency = log_frequency
        self.step = -1
        self.start_time = time.time()

    def begin(self):
        self.step = -1
        self.start_time = time.time()

    def before_run(self, run_context):
        self.step += 1
        return tf.train.SessionRunArgs(self.loss)

    def after_run(self, run_context, run_values):
        if self.step % self.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self.start_time
            self.start_time = current_time

            loss_value = run_values.results[0]
            examples_per_sec = self.log_frequency * self.batch_size / duration
            secs_per_batch = float(duration / self.log_frequency)
            format_str = '{}: step {} | Loss = {:.2f} | ' \
                         '{:.1f} examples/second | {:.3f} seconds/batch)'
            print(format_str.format(datetime.datetime.now(), self.step,
                                    loss_value, examples_per_sec,
                                    secs_per_batch))
