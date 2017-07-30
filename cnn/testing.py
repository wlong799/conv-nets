# coding=utf-8
"""Evaluates model accuracy."""
import datetime
import time

import numpy as np
import tensorflow as tf

import cnn


def evaluate(model_config: cnn.config.ModelConfig, dataset: cnn.input.Dataset):
    """Evaluates accuracy of model.

    Restores variables from the most recent checkpoint file available,
    using their moving averages to obtain better predictions. Outputs the
    percentage of target labels in the test set that are within the top k
    predictions of the model. Validation phase can run repeatedly in
    background if specified, to aid monitoring of training sessions.
    """
    if tf.train.latest_checkpoint(model_config.checkpoints_dir) is None:
        raise RuntimeError("No checkpoints located in '{}'. Cannot run "
                           "evaluation.".format(model_config.checkpoints_dir))
    # Running preprocessing steps on CPU increases performance
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = cnn.compat_utils.get_or_create_global_step()
        model = cnn.model.get_model(model_config.model_name,
                                    model_config.batch_size,
                                    dataset.num_classes)

        images, labels, _ = cnn.input.get_minibatch(
            dataset, model_config.phase, model_config.batch_size,
            model_config.distort_images, model_config.min_example_fraction,
            model_config.num_preprocessing_threads,
            model_config.num_readers, model_config.data_format)

        # Utilize GPU if available for intensive inference step
        device = '/gpu:0' if model_config.num_gpus > 0 else '/cpu:0'
        device_name = 'gpu_0' if model_config.num_gpus > 0 else 'cpu_0'
        with tf.device(device):
            is_training = False
            builder = cnn.model.CNNBuilder(
                images, is_training, model_config.use_batch_norm,
                model_config.weight_decay_rate, model_config.padding_mode,
                model_config.data_format)
            logits = model.inference(builder)
            total_loss = cnn.model.calc_total_loss(
                logits, labels, dataset.class_weights, device_name)

        # Restore from moving averages for better predictions
        variable_averages = tf.train.ExponentialMovingAverage(
            model_config.moving_avg_decay_rate, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        total_examples = dataset.examples_per_epoch(model_config.phase)
        if model_config.phase == 'valid':
            total_examples = int(total_examples *
                                 model_config.bg_valid_set_fraction)
        num_steps = int(total_examples / model_config.batch_size)
        num_examples = num_steps * model_config.batch_size

        top_k_op_dict = {k: tf.nn.in_top_k(logits, labels, k) for k in
                         model_config.top_k_tests}

        if (model_config.phase == 'test' or
                    model_config.bg_valid_repeat_secs == 0):
            session = cnn.monitor.create_testing_session(model_config, saver)
            _eval_once(session, global_step, total_loss, top_k_op_dict,
                       num_steps, num_examples, verbose=True,
                       save_summaries=(model_config.phase == 'valid'))
        else:
            while True:
                session = cnn.monitor.create_testing_session(model_config,
                                                             saver)
                _eval_once(session, global_step, total_loss, top_k_op_dict,
                           num_steps, num_examples, verbose=False,
                           save_summaries=True)
                time.sleep(model_config.bg_valid_repeat_secs)


def _eval_once(session, global_step, loss, top_k_op_dict, num_steps,
               num_examples, verbose, save_summaries):
    """Runs a single evaluation step.

    Args:
        session: Session to run evaluation with.
        global_step: Global step tensor.
        loss: Loss tensor.
        top_k_op_dict: Dictionary providing mapping from int k to the op used
                       for calculating whether prediction of model is in top k.
        num_steps: Number of steps to run evaluation for.
        num_examples: Total number of examples that will have been evaluated.
        verbose: Whether progress should be logged during evaluation.
        save_summaries: Whether summaries should be saved to TensorBoard.
    """
    sorted_k = sorted([key for key in top_k_op_dict])
    num_correct_dict = {k: 0 for k in sorted_k}
    losses = []
    steps = 0
    with session as sess:
        global_step_value = sess.run(global_step)
        while steps < num_steps:
            in_top_k_dict = dict(zip(
                sorted_k,
                sess.run([top_k_op_dict[k] for k in sorted_k])))
            for k in in_top_k_dict:
                num_correct_dict[k] += np.sum(in_top_k_dict[k])
            losses.append(sess.run(loss))

            steps += 1
            if verbose:
                percent_done = steps / num_steps * 100.0
                print("\r>> Evaluating model ({} examples): {:>5.1f}% complete"
                      .format(num_examples, percent_done), end='', flush=True)
        if verbose:
            print()

        precision_dict = {k: num_correct_dict[k] / num_examples for k in
                          num_correct_dict}
        avg_loss = sum(losses) / len(losses)

        summary = tf.Summary()
        summary.value.add(tag='validation/total_loss', simple_value=avg_loss)
        for k in sorted_k:
            summary.value.add(tag='validation/top_{}_precision'.format(k),
                              simple_value=precision_dict[k])
        log = "{}: step {} | Loss = {:.2f}".format(
            datetime.datetime.now(), global_step_value, avg_loss)
        for k in sorted_k:
            log += " | % in Top {:>2} = {:>4.1f}".format(
                k, precision_dict[k] * 100)
        print(log)
        summary_writer = tf.summary.FileWriter('logs/')
        if save_summaries:
            summary_writer.add_summary(summary, global_step_value)
