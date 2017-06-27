# coding=utf-8
"""Evaluates model accuracy."""
import datetime
import math

import numpy as np
import tensorflow as tf

import cnn


def evaluate(model_config: cnn.config.ModelConfig):
    """Evaluates accuracy of model.

    Restores variables from the most recent checkpoint file available,
    using their moving averages (unless specified not to in the model
    configuration) to obtain better predictions. Outputs the percentage of
    target labels in the test set that are within the top k predictions of
    the model, for various values of k specified in the configuration file.

    Args:
        model_config: Model configuration.
    """
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Preprocessing should occur on CPU for improved performance
        with tf.device('/cpu:0'):
            images, labels = cnn.preprocessor.get_minibatch(model_config)

        # Run model
        model = cnn.model.get_model(model_config)
        builder = cnn.model.CNNBuilder(images, model_config)
        logits = model.inference(builder)

        # Set up variable restore using moving averages for better predictions
        if model_config.restore_moving_averages:
            variable_averages = tf.train.ExponentialMovingAverage(
                model_config.ema_decay_rate, global_step, name='var_avg')
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
        else:
            saver = None

        # Set up in_top_k testing for each specified value of k
        top_k_op_dict = {k: tf.nn.in_top_k(logits, labels, k) for k in
                         model_config.top_k_tests}
        num_correct_dict = {k: 0 for k in model_config.top_k_tests}
        num_test_examples = (model_config.examples_per_epoch *
                             model_config.test_set_fraction)
        num_steps = int(math.ceil(num_test_examples / model_config.batch_size))
        total_examples = num_steps * model_config.batch_size
        steps = 0

        # Obtain precisions at each value of k and print results
        time = datetime.datetime.now()
        with cnn.monitor.get_monitored_cnn_session(
                model_config, saver=saver) as mon_sess:
            while not mon_sess.should_stop() and steps < num_steps:
                run_correct_dict = dict(zip(
                    model_config.top_k_tests,
                    mon_sess.run([top_k_op_dict[k] for k in
                                  model_config.top_k_tests])))
                for k in run_correct_dict:
                    num_correct_dict[k] += np.sum(run_correct_dict[k])
                steps += 1
                print("\r{}: Testing on {} examples... {:5.1f}% Complete".
                      format(time, total_examples, steps / num_steps * 100),
                      end='')

        print()
        precision_dict = {k: num_correct_dict[k] / total_examples for k in
                          num_correct_dict}
        for k in precision_dict:
            print("{}: Predictions in Top {:>2} = {:>4.1f}%".format(
                datetime.datetime.now(), k, precision_dict[k] * 100))
