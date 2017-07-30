# coding=utf-8
"""Trains the neural network."""
import re

import tensorflow as tf

import cnn


def train(model_config: cnn.config.ModelConfig, dataset: cnn.input.Dataset):
    """Trains a neural network using the specified configuration and dataset.

    Data preprocessing occurs on the CPU for improved performance. If GPUs
    are available, then inference will run in parallel across all GPUs,
    and average gradients will be computed for each step. Logging,
    checkpoint saving, and TensorBoard visualizations are created using a
    monitored session.
    """
    devices, device_names = _get_devices(model_config.num_gpus)
    num_devices = len(devices)
    # All preprocessing steps should occur on CPU for improved performance
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = cnn.compat_utils.get_or_create_global_step()
        model = cnn.model.get_model(model_config.model_name,
                                    model_config.batch_size,
                                    dataset.num_classes)
        learning_rate = _create_learning_rate(
            model_config.init_learning_rate, model_config.learning_decay_rate,
            dataset.examples_per_epoch(model_config.phase),
            model_config.epochs_per_decay,
            model_config.batch_size * num_devices, global_step)
        optimizer = _create_optimizer(learning_rate, model_config.momentum)

        # Queue provides single access point to examples for all devices
        with tf.name_scope('data_input'):
            images, labels, _ = cnn.input.get_minibatch(
                dataset, model_config.phase, model_config.batch_size,
                model_config.distort_images, model_config.min_example_fraction,
                model_config.num_preprocessing_threads,
                model_config.num_readers, model_config.data_format)
            prefetch_queue = _create_queue([images, labels], 2 * num_devices,
                                           'prefetch_queue')

        device_gradients = []
        with tf.variable_scope(tf.get_variable_scope()):
            is_training = True
            for device, device_name in zip(devices, device_names):
                images, labels = prefetch_queue.dequeue()
                cnn_builder = cnn.model.CNNBuilder(
                    images, is_training, model_config.use_batch_norm,
                    model_config.weight_decay_rate,
                    model_config.padding_mode, model_config.data_format)
                # Place only computationally expensive inference step on GPU
                with tf.device(device), tf.name_scope(device_name):
                    logits = model.inference(cnn_builder)
                    total_loss = cnn.model.calc_total_loss(
                        logits, labels, dataset.class_weights, device_name)
                    average_gradients = optimizer.compute_gradients(total_loss)
                    tf.get_variable_scope().reuse_variables()
                    device_gradients.append(average_gradients)

        _add_activation_summaries(device_names)
        average_loss = _calc_average_loss(device_names)
        average_gradients = _calc_average_gradients(device_gradients)
        apply_grad_op = optimizer.apply_gradients(average_gradients,
                                                  global_step)

        # Storing moving averages allows for better predictions when testing
        with tf.name_scope('moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                model_config.moving_avg_decay_rate, global_step)
            variable_averages_op = variable_averages.apply(
                tf.trainable_variables())

        # update_ops necessary for batch normalization to function properly
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(apply_grad_op, variable_averages_op, *update_ops)
        with cnn.monitor.create_training_session(model_config, average_loss,
                                                 global_step) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def _get_devices(num_gpus):
    """Determine devices that will be used for inference, and give them
    appropriate names for TensorBoard (e.g. /cpu:0 -> cpu_0)."""
    if num_gpus == 0:
        devices = ['/cpu:0']
    else:
        devices = ['/gpu:{}'.format(i) for i in range(num_gpus)]
    device_names = ['_'.join(device[1:].split(':')).lower() for
                    device in devices]
    return devices, device_names


def _create_learning_rate(init_learning_rate, learning_decay_rate,
                          examples_per_epoch, epochs_per_decay,
                          examples_per_step, global_step):
    """Learning rate that exponentially decays as training progresses."""
    with tf.name_scope('learning_rate') as scope:
        steps_per_decay = int(examples_per_epoch * epochs_per_decay /
                              examples_per_step)
        learning_rate = tf.train.exponential_decay(
            init_learning_rate, global_step, steps_per_decay,
            learning_decay_rate, staircase=True, name=scope)
    tf.summary.scalar('{}/value'.format(learning_rate.op.name), learning_rate)
    return learning_rate


def _create_optimizer(learning_rate, momentum):
    """Creates gradient descent optimizer with momentum."""
    return tf.train.MomentumOptimizer(learning_rate, momentum,
                                      name='optimizer')


def _create_queue(tensors, capacity, name):
    """Creates a FIFO queue to hold processed batches for inference."""
    with tf.name_scope(name)as scope:
        queue = tf.FIFOQueue(capacity, [tensor.dtype for tensor in tensors],
                             [tensor.get_shape() for tensor in tensors],
                             name=scope)
        enqueue_op = queue.enqueue(tensors)
        queue_runner = tf.train.QueueRunner(queue, [enqueue_op])
        tf.train.add_queue_runner(queue_runner)
        tf.summary.scalar('fraction_of_{}_full'.format(capacity),
                          queue.size() / capacity)
    return queue


def _calc_average_gradients(device_grads):
    """Calculates average gradient for each variable across all devices.
    Provides TensorBoard summary for each variable and its gradient."""
    with tf.name_scope('average_gradients'):
        average_grads = []
        for grad_and_vars in zip(*device_grads):
            grad_name = grad_and_vars[0][1].op.name
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0, name=grad_name)
            var = grad_and_vars[0][1]
            grad_and_var = (grad, var)
            average_grads.append(grad_and_var)
    for grad, var in average_grads:
        if grad is not None:
            tf.summary.histogram('{}/values'.format(var.op.name), var)
            tf.summary.histogram('{}/gradients'.format(var.op.name), grad)
    return average_grads


def _calc_average_loss(device_names):
    """Calculates average loss across all devices and adds TensorBoard
    summaries."""
    with tf.name_scope('average_loss'):
        average_losses = _average_values_across_devices(tf.GraphKeys.LOSSES,
                                                        device_names)
        regularization_losses = tf.losses.get_regularization_losses()
        model_loss = tf.reduce_sum(average_losses, name='model_loss')
        regularization_loss = tf.reduce_sum(regularization_losses,
                                            name='regularization_loss')
        average_total_loss = tf.add(model_loss, regularization_loss,
                                    name='total_loss')
        for loss in regularization_losses:
            tf.summary.scalar(loss.op.name, loss)
    for loss in average_losses:
        tf.summary.scalar(loss.op.name, loss)
    tf.summary.scalar(model_loss.op.name, model_loss)
    tf.summary.scalar(regularization_loss.op.name, regularization_loss)
    tf.summary.scalar(average_total_loss.op.name, average_total_loss)

    return average_total_loss


def _add_activation_summaries(device_names):
    """For each layer, averages its activation across all devices, and adds
    both a scalar summary of its sparsity, and an overall histogram summary."""
    with tf.name_scope('average_activations'):
        average_activations = _average_values_across_devices(
            tf.GraphKeys.ACTIVATIONS, device_names)
    sparsities = [tf.nn.zero_fraction(
        activation, '{}/sparsity'.format(activation.op.name)) for
        activation in average_activations]
    for activation, sparsity in zip(average_activations, sparsities):
        sparsity_name = '/'.join(sparsity.op.name.split('/')[1:-1])
        tf.summary.scalar(sparsity_name, sparsity)
        activation_name = '/'.join(activation.op.name.split('/')[2:])
        tf.summary.histogram(activation_name, activation)


def _average_values_across_devices(collection_name, device_names):
    all_values = [tf.get_collection(collection_name, device_name) for
                  device_name in device_names]
    average_values = []
    for value_across_all_devices in zip(*all_values):
        name = re.sub('[gc]pu_[0-9]+', collection_name,
                      value_across_all_devices[0].op.name)
        with tf.name_scope(name) as scope:
            expanded_values = []
            for device_value in value_across_all_devices:
                expanded_values.append(tf.expand_dims(device_value, 0))
            average_value = tf.reduce_mean(tf.concat(expanded_values, 0), 0,
                                           name=scope)
            average_values.append(average_value)
    return average_values
