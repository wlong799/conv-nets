# coding=utf-8
"""
Runs model for training the neural network on the CPU. Creates the training
step and runs it.
"""
import tensorflow as tf
import cnn


def train(batch_size=32):
    """ Builds an optimizer to update parameters of model."""
    with tf.device('/cpu:0'):
        images, labels = cnn.preprocessor.get_minibatch(
            'data/cifar10', 64, 24, 24)
    logits = cnn.inference.inference(images)
    total_loss = cnn.inference.loss(logits, labels)
    opt = tf.train.GradientDescentOptimizer(0.001)
    train_op = opt.minimize(total_loss)
    with tf.train.MonitoredTrainingSession(
            hooks=[tf.train.StopAtStepHook(last_step=1000),
                   tf.train.NanTensorHook(total_loss)]) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)


if __name__ == '__main__':
    train()
