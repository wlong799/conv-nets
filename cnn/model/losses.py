# coding=utf-8
"""Loss calculation for model."""
import tensorflow as tf


def calc_total_loss(logits, labels, class_weights, device_scope):
    """Calculates the cross entropy loss for the given device, scaled by
    values in class_weights, if provided. Adds any other losses (e.g.
    regularization losses) to calculate the total loss on the device."""
    with tf.name_scope('total_loss') as scope:
        if class_weights is not None:
            weights = [class_weights[label] for label in labels]
        else:
            weights = 1.0
        tf.losses.sparse_softmax_cross_entropy(labels, logits, weights)

        losses = tf.losses.get_losses(device_scope)
        regularization_losses = tf.losses.get_regularization_losses()
        all_losses = losses + regularization_losses
        total_loss = tf.add_n(all_losses, scope)

    return total_loss
