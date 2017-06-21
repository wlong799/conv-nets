# coding=utf-8
"""
Runs model for training the neural network on the CPU. Creates the training
step and runs it.
"""
import tensorflow as tf
import cnn

BATCH_SIZE = 32
IMAGE_SIZE = 24
DISTORT = True
DATA_FORMAT = 'NHWC'
NUM_PREPROCESS_THREADS = 32
MIN_BUFFER_SIZE = 10000
LEARNING_RATE = 0.1


def train():
    """ Builds an optimizer to update parameters of model."""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # Preprocessing should occur on CPU for improved performance
        with tf.device('/cpu:0'):
            images, labels = cnn.preprocessor.get_minibatch(
                'data/cifar10', BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE,
                distort_images=DISTORT, data_format=DATA_FORMAT,
                num_threads=NUM_PREPROCESS_THREADS,
                min_buffer_size=MIN_BUFFER_SIZE)

        logits = cnn.inference.inference(images)
        total_loss = cnn.inference.loss(logits, labels)

        opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        grads = opt.compute_gradients(total_loss)

        apply_grad_op = opt.apply_gradients(grads, global_step)

        with tf.train.MonitoredTrainingSession(
                hooks=[tf.train.StopAtStepHook(last_step=1000),
                       tf.train.NanTensorHook(total_loss)]) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(apply_grad_op)


if __name__ == '__main__':
    train()
