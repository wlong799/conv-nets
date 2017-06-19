# coding=utf-8
"""
Runs model for training the neural network on the CPU. Creates the training
step and runs it.
"""
import tensorflow as tf
import cnn


def train():
    """ Builds an optimizer to update parameters of model."""
    with tf.device('/cpu:0'):
        image_batch, label_batch = cnn.data_reader.get_input_batch(
            'data', 256)
    logits = cnn.model.inference(image_batch)
    total_loss = cnn.model.loss(logits, label_batch)
    opt = tf.train.GradientDescentOptimizer(0.001)
    train_op = opt.minimize(total_loss)
    with tf.train.MonitoredTrainingSession(
            hooks=[tf.train.StopAtStepHook(last_step=1000),
                   tf.train.NanTensorHook(total_loss)]) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)


if __name__ == '__main__':
    train()
