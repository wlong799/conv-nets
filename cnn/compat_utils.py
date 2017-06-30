# coding=utf-8
"""Utilities to provide compatibility through TensorFlow 1.0.

The global step functions below are included as a part of TensorFlow 1.2,
but are not provided in TensorFlow 1.0. Until HPCs are updated to 1.2,
it is necessary to use these functions instead, which were altered directly
from the 1.2 source code found at:
https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python
/training/training_util.py."""
import tensorflow as tf


def get_or_create_global_step(graph=None):
    """Replaces tf.get_or_create_global_step()"""
    graph = graph or tf.get_default_graph()
    global_step_tensor = _get_global_step(graph)
    if global_step_tensor is None:
        global_step_tensor = _create_global_step(graph)
    return global_step_tensor


def _assert_global_step(global_step_tensor):
    if not (isinstance(global_step_tensor, tf.Variable) or
                isinstance(global_step_tensor, tf.Tensor)):
        raise TypeError(
            "Existing 'global_step' must be a Variable or Tensor: {}.".format(
                global_step_tensor))
    if not global_step_tensor.dtype.base_dtype.is_integer:
        raise TypeError(
            "Existing 'global_step' does not have integer type: {}.".format(
                global_step_tensor.dtype))
    if global_step_tensor.get_shape().ndims != 0:
        raise TypeError("Existing 'global_step' is not scalar: {}.".format(
            global_step_tensor.get_shape()))


def _create_global_step(graph=None):
    graph = graph or tf.get_default_graph()
    if _get_global_step(graph) is not None:
        raise ValueError("'global_step' already exists.")
    with graph.as_default() as g, g.name_scope(None):
        return tf.get_variable(
            tf.GraphKeys.GLOBAL_STEP,
            shape=[],
            dtype=tf.int64,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                         tf.GraphKeys.GLOBAL_STEP])


def _get_global_step(graph=None):
    graph = graph or tf.get_default_graph()
    global_step_tensors = graph.get_collection(tf.GraphKeys.GLOBAL_STEP)
    if len(global_step_tensors) == 1:
        global_step_tensor = global_step_tensors[0]
    elif not global_step_tensors:
        try:
            global_step_tensor = graph.get_tensor_by_name('global_step:0')
        except KeyError:
            return None
    else:
        tf.logging.error("Multiple tensors in global_step collection.")
        return None
    _assert_global_step(global_step_tensor)
    return global_step_tensor
