# coding=utf-8
"""Builds a convolutional neural network for classifying CIFAR-10 images.

Methods in this module build a deep CNN for predicting image classes from the
CIFAR-10 dataset, as well as calculate the loss of the model and the
training operation necessary for updating the parameters.

Model architecture employs multiple layers of convolution/pooling/local
response normalization layers, with rectified linear activation following
each convolution layer. It is followed by fully connected layers
regularized through dropout, and a final softmax layer with 10 units for the
10 classes.

Loss is calculated using cross-entropy between the labels and the softmax
activations of the final layer, as well as a weight decay term to regularize
the model for better generalization.

Training is performed using a simple gradient descent method with
exponential decay of the learning rate.
"""
import tensorflow as tf


def cpu_variable(name, shape, initializer):
    """Creates a variable on CPU"""
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, tf.float32, initializer)
    return var


def weight_decay_variable(name, shape, stddev, decay_const):
    """Creates a variable"""
    var = cpu_variable(name, shape, tf.truncated_normal_initializer(
        stddev=stddev))
    weight_decay = tf.multiply(tf.nn.l2_loss(var), decay_const,
                               name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var


def inference(images):
    """Deep CNN model to make class predictions from CIFAR-10 images.

    Architecture is as follows:
    conv3x3x64/1 -> conv3x3x64/1 -> norm -> pool3x3/2 -> conv5x5x128/1 -> norm
        -> pool3x3/2 -> fc512 -> fc256 -> linear10

    ReLU is applied after each convolutional layer and fully connected layer.

    Choices made for better regularization are as follows. Pooling layers have
    slight overlap (stride of 2x2). Both fully connected layers employ
    drop out with probability of 0.5. And weight decay is applied to  all
    weights in the model.

    Args:
        images: 4D Tensor [BATCH_SIZE, PROCESSED_IMAGE_DIM*]. The
                batch of images obtained from get_input_batch().

    Returns:
        logits: 2D Tensor [BATCH_SIZE, 10].
    """
    batch_size = images.get_shape().as_list()[0]
    # conv1
    with tf.variable_scope('conv1') as scope:
        filter = cpu_variable('weights', [3, 3, 3, 64], 5e-2)
        conv = tf.nn.conv2d(images, filter, [1, 1, 1, 1], padding='SAME')
        biases = cpu_variable('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # conv2
    with tf.variable_scope('conv2') as scope:
        filter = cpu_variable('weights', [3, 3, 64, 64], 5e-2)
        conv = tf.nn.conv2d(conv1, filter, [1, 1, 1, 1], padding='SAME')
        biases = cpu_variable('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # normalization/pooling 1
    norm1 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # conv3
    with tf.variable_scope('conv3') as scope:
        filter = cpu_variable('weights', [5, 5, 64, 128], 5e-2)
        conv = tf.nn.conv2d(pool1, filter, [1, 1, 1, 1], padding='SAME')
        biases = cpu_variable('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    # normalization/pooling 2
    norm2 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    # fc1
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pool2, [pool2.get_shape()[0].value, -1])
        dim = reshape.get_shape()[1].value
        weights = weight_decay_variable('weights', [dim, 512], 5e-2, 5e-3)
        biases = cpu_variable('biases', [512], tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # fc2
    with tf.variable_scope('fc1') as scope:
        weights = weight_decay_variable('weights', [512, 256], 5e-2, 5e-3)
        biases = cpu_variable('biases', [256], tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

    # linear
    with tf.variable_scope('linear') as scope:
        weights = cpu_variable('weights', [192, 10], 1 / 192.0)
        biases = cpu_variable('biases', [10], tf.constant_initializer(0.0))
        linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return linear


def loss(logits, labels):
    """Calculates total loss including cross-entropy and weight decay terms.

    Args:
        logits: 2D Tensor [BATCH_SIZE, 10]. Logits calculated by inference().
        labels: 1D Tensor [BATCH_SIZE]. Single label of correct class for
        each image

    Returns:
        Loss for the current model.
    """
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')
