# coding=utf-8
"""
Runs model for training the neural network on the CPU. Creates the training
step and runs it.
"""
import tensorflow as tf
import cnn


def train(model_name, image_height, image_width, num_classes, data_dir,
          checkpoints_dir=None, summaries_dir=None, batch_size=128,
          distort_images=True, num_preprocess_threads=32,
          min_buffer_size=10000, learning_rate=0.1, log_frequency=10,
          save_checkpoint_secs=600, save_summaries_steps=100):
    """ Builds an optimizer to update parameters of model."""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # Preprocessing should occur on CPU for improved performance
        with tf.device('/cpu:0'):
            images, labels = cnn.preprocessor.get_minibatch(
                data_dir, batch_size, image_height, image_width,
                distort_images=distort_images,
                num_threads=num_preprocess_threads,
                min_buffer_size=min_buffer_size)

        model = cnn.model.get_model(model_name, batch_size, num_classes)
        builder = cnn.builder.CNNBuilder(images, 3, True)
        logits = model.inference(builder)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='cross_entropy')

        opt = tf.train.GradientDescentOptimizer(learning_rate)
        grads = opt.compute_gradients(cross_entropy_mean)

        apply_grad_op = opt.apply_gradients(grads, global_step)

        with cnn.monitor.get_monitored_cnn_session(
                checkpoints_dir, summaries_dir, cross_entropy_mean,
                batch_size, log_frequency, save_checkpoint_secs,
                save_summaries_steps) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(apply_grad_op)
