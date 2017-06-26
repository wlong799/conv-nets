# coding=utf-8
"""
Runs model for training the neural network on the CPU. Creates the training
step and runs it.
"""
import tensorflow as tf

import cnn


def train(model_config: cnn.config.ModelConfig):
    """ Builds an optimizer to update parameters of model."""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # Preprocessing should occur on CPU for improved performance
        with tf.device('/cpu:0'):
            images, labels = cnn.preprocessor.get_minibatch(model_config)

        model = cnn.model.get_model(model_config)
        builder = cnn.model.CNNBuilder(images, model_config)
        logits = model.inference(builder)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='cross_entropy')

        opt = tf.train.GradientDescentOptimizer(model_config.learning_rate)
        grads = opt.compute_gradients(cross_entropy_mean)

        apply_grad_op = opt.apply_gradients(grads, global_step)

        with cnn.monitor.get_monitored_cnn_session(
                cross_entropy_mean, model_config) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(apply_grad_op)
