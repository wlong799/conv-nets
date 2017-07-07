# coding=utf-8
"""Evaluates model accuracy."""
import datetime
import time

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
    In validation phase, model runs repeatedly in background by default,
    to be able to run in parallel with training. In testing phase,
    model just runs once and outputs results.

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

        # Preprocess on CPU for improved performance
        with tf.device('/cpu:0'):
            images, labels = cnn.input.get_minibatch(
                dataset, model_config.phase, model_config.batch_size,
                model_config.distort_images, model_config.min_example_fraction,
                model_config.num_preprocessing_threads,
                model_config.data_format)

        # Run inference and calculate loss
        image_channels = dataset.image_shape[-1]
        builder = cnn.model.CNNBuilder(
            images, image_channels, False,
            model_config.weight_decay_rate, model_config.padding_mode,
            model_config.data_format, model_config.data_type)
        logits = model.inference(builder)
        loss = cnn.training.calc_total_loss(logits, labels)

        # Set up variable restore using moving averages for better predictions
        if model_config.restore_moving_averages:
            variable_averages = tf.train.ExponentialMovingAverage(
                model_config.moving_avg_decay_rate, global_step,
                name='var_avg')
            variables_to_restore = variable_averages.variables_to_restore()
        else:
            variables_to_restore = None

        # Set up in_top_k testing for each specified value of k
        top_k_op_dict = {k: tf.nn.in_top_k(logits, labels, k) for k in
                         model_config.top_k_tests}

        # Determine number of examples to run
        if model_config.phase == 'valid':
            num_steps = int(dataset.examples_per_epoch(model_config.phase) *
                            model_config.valid_set_fraction /
                            model_config.batch_size)
        else:
            num_steps = int(dataset.examples_per_epoch(model_config.phase) /
                            model_config.batch_size)
        num_examples = num_steps * model_config.batch_size

        # If testing, run once; if validation, run repeatedly in background
        if model_config.phase == 'test' or model_config.valid_repeat_secs == 0:
            session = cnn.monitor.create_testing_session(
                model_config, variables_to_restore)
            _eval_once(session, global_step, loss,
                       top_k_op_dict, num_steps, num_examples, True,
                       model_config.phase == 'valid')
        else:
            while True:
                session = cnn.monitor.create_testing_session(
                    model_config, variables_to_restore)
                _eval_once(session, global_step, loss,
                           top_k_op_dict, num_steps, num_examples, False, True)
                time.sleep(model_config.valid_repeat_secs)


def _eval_once(session, global_step, loss, top_k_op_dict, num_steps,
               num_examples, verbose, save_summaries):
    top_k_tests = sorted([key for key in top_k_op_dict])
    num_correct_dict = {k: 0 for k in top_k_tests}
    losses = []
    steps = 0
    with session as sess:
        global_step_value = sess.run(global_step)
        while steps < num_steps:
            run_correct_dict = dict(zip(
                top_k_tests,
                sess.run([top_k_op_dict[k] for k in top_k_tests])))
            for k in run_correct_dict:
                num_correct_dict[k] += np.sum(run_correct_dict[k])
            losses.append(sess.run(loss))
            steps += 1
            if verbose and steps % 100 == 0:
                percent_done = steps / num_steps * 100.0
                print("\r>> Evaluating model ({} examples): {:>5.1f}% complete"
                      .format(num_examples, percent_done), end='')
        precision_dict = {k: num_correct_dict[k] / num_examples for k in
                          num_correct_dict}
        avg_loss = sum(losses) / len(losses)
        summary = tf.Summary()
        log = "{}: step {} | Loss = {:.2f}".format(
            datetime.datetime.now(), global_step_value, avg_loss)
        summary.value.add(tag='validation/total_loss', simple_value=avg_loss)
        for k in precision_dict:
            log += " | % in Top {:>2} = {:>4.1f}".format(
                k, precision_dict[k] * 100)
            summary.value.add(tag='validation/top_{}_precision'.format(k),
                              simple_value=precision_dict[k])
        print(log)
        summary_writer = tf.summary.FileWriter('logs/')
        if save_summaries:
            summary_writer.add_summary(summary, global_step_value)
