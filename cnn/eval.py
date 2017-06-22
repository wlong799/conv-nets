# coding=utf-8
"""Evaluates the accuracy of a model by loading a recent checkpoint."""
import datetime
import math

import numpy as np
import tensorflow as tf

import cnn


def evaluate(model_name, image_height, image_width, num_classes, data_dir,
             checkpoints_dir, num_examples=1000, data_format='NHWC',
             batch_size=128,
             num_preprocess_threads=32, min_buffer_size=10000, config=None):
    """Evaluates most recent checkpoint for accuracy."""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # Preprocessing should occur on CPU for improved performance
        with tf.device('/cpu:0'):
            images, labels = cnn.preprocessor.get_minibatch(
                data_dir, batch_size, image_height, image_width, 'test',
                data_format=data_format, num_threads=num_preprocess_threads,
                min_buffer_size=min_buffer_size)

        model = cnn.model.get_model(model_name, batch_size, num_classes)
        builder = cnn.builder.CNNBuilder(images, 3, False,
                                         data_format=data_format)
        logits = model.inference(builder)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        num_steps = int(math.ceil(num_examples / batch_size))
        total_examples = num_steps * batch_size
        steps = 0
        num_correct = 0

        with cnn.monitor.get_monitored_cnn_session(
                checkpoints_dir, log_frequency=None, save_checkpoint_secs=None,
                save_summaries_steps=None, config=config) as mon_sess:
            while not mon_sess.should_stop() and steps < num_steps:
                predictions = mon_sess.run(top_k_op)
                num_correct += np.sum(predictions)
                steps += 1

        precision = num_correct / total_examples
        print("{}: Precision @ 1 = {:.3f}".format(datetime.datetime.now(),
                                                  precision))

