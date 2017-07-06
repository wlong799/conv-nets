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
        # Set up model
        global_step = cnn.compat_utils.get_or_create_global_step()
        dataset = cnn.input.get_dataset(model_config.dataset_name,
                                        model_config.data_dir,
                                        model_config.overwrite)
        model = cnn.model.get_model(model_config.model_type,
                                    model_config.batch_size,
                                    dataset.num_classes)

        # Preprocessing should occur on CPU for improved performance
        with tf.device('/cpu:0'):
            images, labels = cnn.input.get_minibatch(
                dataset, model_config.phase, model_config.batch_size,
                model_config.distort_images, model_config.min_example_fraction,
                model_config.num_preprocessing_threads,
                model_config.data_format)

        builder = cnn.model.CNNBuilder(
            images, dataset.image_shape[-1], False,
            model_config.weight_decay_rate, model_config.padding_mode,
            model_config.data_format, model_config.data_type)
        logits = model.inference(builder)

        # Set up variable restore using moving averages for better predictions
        if model_config.restore_moving_averages:
            variable_averages = tf.train.ExponentialMovingAverage(
                model_config.moving_avg_decay_rate, global_step,
                name='var_avg')
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
        else:
            saver = None
        # Set up in_top_k testing for each specified value of k
        top_k_op_dict = {k: tf.nn.in_top_k(logits, labels, k) for k in
                         model_config.top_k_tests}

        # Obtain precisions at each value of k and print results
        # time = datetime.datetime.now()
        while True:
            num_steps = int(dataset.examples_per_epoch(model_config.phase) *
                            model_config.eval_set_fraction /
                            model_config.batch_size)
            num_examples = num_steps * model_config.batch_size
            session = cnn.monitor.get_monitored_cnn_session(model_config,
                                                            saver=saver)
            _eval_once(top_k_op_dict, num_steps, num_examples, session)


def _eval_once(top_k_op_dict, num_steps, num_examples, session):
    top_k_tests = [key for key in top_k_op_dict]
    num_correct_dict = {k: 0 for k in top_k_tests}
    steps = 0
    with session as mon_sess:
        while not mon_sess.should_stop() and steps < num_steps:
            run_correct_dict = dict(zip(
                top_k_tests,
                mon_sess.run([top_k_op_dict[k] for k in
                              top_k_tests])))
            for k in run_correct_dict:
                num_correct_dict[k] += np.sum(run_correct_dict[k])
            steps += 1
    precision_dict = {k: num_correct_dict[k] / num_examples for k in
                      num_correct_dict}
    for k in precision_dict:
        print("{}: Predictions in Top {:>2} = {:>4.1f}%".format(
            datetime.datetime.now(), k, precision_dict[k] * 100))
