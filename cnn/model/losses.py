# coding=utf-8
"""Loss calculation for model."""
import tensorflow as tf


def calc_total_loss(logits, labels, class_weights=None, device_scope=None):
    """Calculates cross entropy + regularization loss. Weights the cross
    entropy calculations by the weights in class_weights, if provided. If
    device_scope is provided, calculates the losses only for that device."""
    if class_weights is not None:
        weights = [class_weights[label] for label in labels]
    else:
        weights = 1.0
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels, logits, weights, device_scope)
    regularization_loss = tf.losses.get_regularization_loss(device_scope)
    return tf.add(loss, regularization_loss)
