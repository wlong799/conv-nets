# coding=utf-8
"""Trains the neural network."""
import re

import tensorflow as tf

import cnn


def train(model_config: cnn.config.ModelConfig, dataset: cnn.input.Dataset):
    """Trains a neural network with the specified configuration.

    Model trains a neural network. Preprocessing and all variable storage
    occurs on the CPU. If GPUs are available, then inference will run in
    parallel across all GPUs, and average gradients will be computed for each
    step. Logging, checkpoint saving, and TensorBoard visualizations are
    created using a monitored session.
    """
    devices, device_names = _get_devices(model_config.num_gpus)
    num_devices = len(devices)
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Initialize model-wide variables
        global_step = cnn.compat_utils.get_or_create_global_step()
        model = cnn.model.get_model(
            model_config.model_name, model_config.batch_size,
            dataset.num_classes)
        optimizer = _create_optimizer(
            model_config, dataset.examples_per_epoch(model_config.phase),
            global_step)

        # Set up prefetch queue for examples to be accessible by all devices
        with tf.name_scope('data_input'):
            images, labels, _ = cnn.input.get_minibatch(
                dataset, model_config.phase, model_config.batch_size,
                model_config.distort_images, model_config.min_example_fraction,
                model_config.num_preprocessing_threads,
                model_config.num_readers, model_config.data_format)
            prefetch_queue = _create_queue([images, labels], 2 * num_devices,
                                           'prefetch_queue')
        # Run inference for a batch on each available device in parallel
        device_gradients = []
        total_losses = []
        with tf.variable_scope(tf.get_variable_scope()):
            is_training = True
            for device, device_name in zip(devices, device_names):
                images, labels = prefetch_queue.dequeue()
                cnn_builder = cnn.model.CNNBuilder(
                    images, is_training, model_config.use_batch_norm,
                    model_config.weight_decay_rate,
                    model_config.padding_mode, model_config.data_format)
                # Place only computationally expensive inference step on device
                with tf.device(device), tf.name_scope(device_name) as scope:
                    logits = model.inference(cnn_builder)
                    total_loss = cnn.losses.calc_total_loss(
                        logits, labels, dataset.class_weights, scope)
                    gradients = optimizer.compute_gradients(total_loss)
                    tf.get_variable_scope().reuse_variables()
                    device_gradients.append(gradients)
                    total_losses.append(total_loss)

        # Average gradients across all devices and apply to variables
        gradients = _calc_average_gradients(device_gradients)
        apply_grad_op = optimizer.apply_gradients(gradients, global_step)

        # Add summaries
        _add_loss_summaries(device_names)
        _add_activation_summaries(device_names)
        total_loss = tf.reduce_mean(total_losses, name='total_loss')
        for grad, var in gradients:
            if grad is not None:
                tf.summary.histogram('{}/values'.format(var.op.name), var)
                tf.summary.histogram('{}/gradients'.format(var.op.name), grad)

        # Track variable moving averages for better predictions
        with tf.name_scope('moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                model_config.moving_avg_decay_rate, global_step,
                name='moving_avg')
            variable_averages_op = variable_averages.apply(
                tf.trainable_variables())

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(apply_grad_op, variable_averages_op, *update_ops)
        with cnn.monitor.create_training_session(model_config, total_loss,
                                                 global_step) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def _get_devices(num_gpus):
    """Determine devices that will be used for inference, and give them
    appropriate names for TensorBoard (e.g. /cpu:0 -> CPU_0). """
    if num_gpus == 0:
        devices = ['/cpu:0']
    else:
        devices = ['/gpu:{}'.format(i) for i in range(num_gpus)]
    device_names = ['_'.join(device[1:].split(':')).upper() for
                    device in devices]
    return devices, device_names


def _create_optimizer(model_config, examples_per_epoch, global_step):
    """Creates exponentially decaying optimizer for training model."""
    steps_per_decay = int(examples_per_epoch *
                          model_config.epochs_per_decay /
                          model_config.batch_size)
    learning_rate = tf.train.exponential_decay(
        model_config.init_learning_rate, global_step, steps_per_decay,
        model_config.learning_decay_rate, True, 'learning_rate')
    tf.summary.scalar(learning_rate.op.name, learning_rate)
    # TODO: Look into using other optimizer types
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer


def _create_queue(tensors, capacity, name):
    """Creates a FIFO queue for holding processed batches for inference."""
    queue = tf.FIFOQueue(capacity, [tensor.dtype for tensor in tensors],
                         [tensor.get_shape() for tensor in tensors], name=name)
    enqueue_op = queue.enqueue(tensors)
    queue_runner = tf.train.QueueRunner(queue, [enqueue_op])
    tf.train.add_queue_runner(queue_runner)
    tf.summary.scalar('{}/fraction_of_{}_full'.format(name, capacity),
                      queue.size() / capacity)
    return queue


def _calc_average_gradients(device_grads):
    with tf.name_scope('average_gradients'):
        average_grads = []
        for grad_and_vars in zip(*device_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


def _add_loss_summaries(device_names):
    """Averages losses across all devices and adds summary for each."""
    with tf.name_scope('loss_summary'):
        average_losses = _average_values_across_devices(
            tf.GraphKeys.LOSSES, device_names)
        average_total_loss = tf.reduce_sum(average_losses,
                                           name='losses/total_loss')
    for loss in average_losses + [average_total_loss]:
        stripped_name = '/'.join(loss.op.name.split('/')[1:])
        tf.summary.scalar(stripped_name, loss)


def _add_activation_summaries(device_names):
    """For each layer, averages its activation across all devices, and adds
    both a scalar summary of its sparsity, and an overall histogram summary."""
    # Use name scope for calculation ops for cleaner graph view
    with tf.name_scope('activation_summary'):
        average_activations = _average_values_across_devices(
            tf.GraphKeys.ACTIVATIONS, device_names)
        sparsities = [tf.nn.zero_fraction(activation) for
                      activation in average_activations]
    # Add summaries, with names stripped of prefix for cleaner summary view
    for activation, sparsity in zip(average_activations, sparsities):
        stripped_name = '/'.join(activation.op.name.split('/')[1:])
        tf.summary.scalar('{}/sparsity'.format(stripped_name), sparsity)
        stripped_name = '/'.join(stripped_name.split('/')[1:])
        tf.summary.histogram('{}/values'.format(stripped_name), activation)


def _average_values_across_devices(collection_name, device_names):
    all_values = [tf.get_collection(collection_name, device_name) for
                  device_name in device_names]
    average_values = []
    for value_across_all_devices in zip(*all_values):
        # Strip device prefix from name and replace with collection name
        name = re.sub('[GC]PU_[0-9]+', collection_name,
                      value_across_all_devices[0].op.name)
        expanded_values = []
        for device_value in value_across_all_devices:
            expanded_values.append(tf.expand_dims(device_value, 0))
        average_value = tf.reduce_mean(tf.concat(expanded_values, 0), 0,
                                       name=name)
        average_values.append(average_value)
    return average_values