# coding=utf-8
"""Loss calculation for model."""
import tensorflow as tf


def calc_total_loss(logits, labels, class_weights=None, device_scope=None):
    """Calculates cross entropy + weight decay loss. Weights the cross
    entropy calculations by the weights in class_weights, if provided. If
    device_scope is provided, calculates the losses only for that device."""
    if class_weights:
        targets = tf.one_hot(labels, len(class_weights))
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
            targets=targets, logits=logits, pos_weight=class_weights,
            name='cross_entropy_per_example')
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses', device_scope), 'total_loss')
