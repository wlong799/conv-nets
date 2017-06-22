# coding=utf-8
"""Custom session monitor for CNNs"""
import datetime
import time

import tensorflow as tf


def get_monitored_cnn_session(checkpoints_dir=None, summaries_dir=None,
                              loss=None, batch_size=None, log_frequency=10,
                              save_checkpoint_secs=600,
                              save_summaries_steps=100, config=None):
    checkpoints_dir = checkpoints_dir or 'checkpoints/'
    scaffold = tf.train.Scaffold()
    session_creator = tf.train.ChiefSessionCreator(
        scaffold, config=config, checkpoint_dir=checkpoints_dir)

    hooks = []
    if checkpoints_dir and save_checkpoint_secs and save_checkpoint_secs > 0:
        hooks.append(tf.train.CheckpointSaverHook(checkpoints_dir,
                                                  save_checkpoint_secs,
                                                  scaffold=scaffold))
    if summaries_dir and save_summaries_steps and save_summaries_steps > 0:
        hooks.append(tf.train.StepCounterHook(save_summaries_steps,
                                              output_dir=summaries_dir))
        hooks.append(tf.train.SummarySaverHook(save_summaries_steps,
                                               output_dir=summaries_dir,
                                               scaffold=scaffold))
    if loss is not None:
        hooks.append(tf.train.NanTensorHook(loss))
        if (batch_size and batch_size > 0) and (
                    log_frequency and log_frequency > 0):
            hooks.append(_LoggerHook(loss, batch_size, log_frequency))

    return tf.train.MonitoredSession(session_creator, hooks)


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
