# coding=utf-8
"""Trains the neural network."""
import re

import tensorflow as tf

import cnn
import cnn.model.implementations.model_selection


def train(model_config: cnn.config.ModelConfig):
    """Trains a neural network with the specified configuration.

    Model trains a neural network. Preprocessing and all variable storage
    occurs on the CPU. If GPUs are available, then inference will run in
    parallel across all GPUs, and average gradients will be computed for each
    step. Logging, checkpoint saving, and TensorBoard visualizations are
    created as well using a monitored session.

    Args:
        model_config: Model configuration.
    """
    # Determine devices to use for inference.
    devices = _get_devices(model_config.num_gpus)
    num_devices = len(devices)
    device_names = ['_'.join(device[1:].split(':')).upper() for
                    device in devices]  # /cpu:0 -> CPU_0, /gpu:1 -> GPU_1, etc
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Initialize model-wide variables
        global_step = tf.train.get_or_create_global_step()
        optimizer = _create_optimizer(model_config, global_step)
        model = cnn.model.implementations.model_selection.get_model(
            model_config.model_type, model_config.batch_size,
            model_config.num_classes)

        # Set up prefetch queue for examples to be accessible by all devices
        with tf.name_scope('data_input'):
            images, labels = cnn.preprocessor.get_minibatch(model_config)
            prefetch_queue = _create_queue([images, labels], 2 * num_devices,
                                           'prefetch_queue')

        # Run inference for a batch on each available device in parallel
        device_gradients = []
        total_losses = []
        with tf.variable_scope(tf.get_variable_scope()):
            for device, device_name in zip(devices, device_names):
                images, labels = prefetch_queue.dequeue()
                cnn_builder = cnn.model.CNNBuilder(
                    images, model_config.image_channels, True,
                    model_config.weight_decay_rate,
                    model_config.padding_mode, model_config.data_format,
                    model_config.data_type)
                # Place only computationally expensive inference step on device
                with tf.device(device), tf.name_scope(device_name) as scope:
                    logits = model.inference(cnn_builder)
                    total_loss = _calc_total_loss(logits, labels, scope)
                    gradients = optimizer.compute_gradients(total_loss)
                    tf.get_variable_scope().reuse_variables()
                    device_gradients.append(gradients)
                    total_losses.append(total_loss)

        _add_loss_summaries(device_names)
        _add_activation_summaries(device_names)
        # Average gradients across all devices and apply to variables
        gradients = _calc_average_gradients(device_gradients)
        apply_grad_op = optimizer.apply_gradients(gradients, global_step)
        for grad, var in gradients:
            if grad is not None:
                tf.summary.histogram('{}/values'.format(var.op.name), var)
                tf.summary.histogram('{}/gradients'.format(var.op.name), grad)

        # Track variable moving averages for better predictions
        variable_averages = tf.train.ExponentialMovingAverage(
            model_config.moving_avg_decay_rate, global_step, name='var_avg')
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables())

        total_loss = tf.reduce_mean(total_losses)
        train_op = tf.group(apply_grad_op, variable_averages_op)
        with cnn.monitor.get_monitored_cnn_session(model_config, total_loss,
                                                   global_step) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def _calc_average_gradients(device_grads):
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


def _get_devices(num_gpus):
    if num_gpus == 0:
        return ['/cpu:0']
    else:
        return ['/gpu:{}'.format(i) for i in range(num_gpus)]


def _create_optimizer(model_config, global_step):
    steps_per_decay = int(model_config.examples_per_epoch *
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
    queue = tf.FIFOQueue(capacity, [tensor.dtype for tensor in tensors],
                         [tensor.get_shape() for tensor in tensors], name=name)
    enqueue_op = queue.enqueue(tensors)
    queue_runner = tf.train.QueueRunner(queue, [enqueue_op])
    tf.train.add_queue_runner(queue_runner)
    tf.summary.scalar('{}/fraction_of_{}_full'.format(name, capacity),
                      queue.size() / capacity)
    return queue


def _calc_total_loss(logits, labels, scope=None):
    # Calculate cross entropy loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses', scope), 'total_loss')


def _add_loss_summaries(device_names):
    average_losses = _average_values_across_devices('losses', device_names)
    average_total_loss = tf.reduce_sum(average_losses,
                                       name='losses/total_loss')
    for loss in average_losses + [average_total_loss]:
        tf.summary.scalar(loss.op.name, loss)


def _add_activation_summaries(device_names):
    average_activations = _average_values_across_devices('activations',
                                                         device_names)
    for activation in average_activations:
        tf.summary.histogram('{}/values'.format(activation.op.name),
                             activation)
        tf.summary.scalar('{}/sparsity'.format(activation.op.name),
                          tf.nn.zero_fraction(activation))


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
