# coding=utf-8
"""Evaluates the accuracy of a model by loading a recent checkpoint."""
import datetime
import math

import numpy as np
import tensorflow as tf

import cnn


def evaluate(model_config: cnn.config.ModelConfig):
    """Evaluates most recent checkpoint for accuracy."""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # Preprocessing should occur on CPU for improved performance
        with tf.device('/cpu:0'):
            images, labels = cnn.preprocessor.get_minibatch(model_config)

        model = cnn.model.get_model(model_config)
        builder = cnn.model.CNNBuilder(images, model_config)
        logits = model.inference(builder)

        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_3_op = tf.nn.in_top_k(logits, labels, 3)

        num_steps = int(math.ceil(
            model_config.examples_per_epoch / model_config.batch_size))
        total_examples = num_steps * model_config.batch_size
        steps = 0
        num_correct_1 = 0
        num_correct_3 = 0

        with cnn.monitor.get_monitored_cnn_session(
                None, model_config) as mon_sess:
            while not mon_sess.should_stop() and steps < num_steps:
                predictions_1, predictions_3 = mon_sess.run(
                    [top_1_op, top_3_op])
                num_correct_1 += np.sum(predictions_1)
                num_correct_3 += np.sum(predictions_3)
                steps += 1

        precision_1 = num_correct_1 / total_examples
        precision_3 = num_correct_3 / total_examples
        print("{}: Precision @ 1 = {:.3f}".format(datetime.datetime.now(),
                                                  precision_1))
        print("{}: Precision @ 3 = {:.3f}".format(datetime.datetime.now(),
                                                  precision_3))
