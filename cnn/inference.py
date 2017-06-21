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
import cnn


def inference(images):
    """Deep CNN model to make class predictions from CIFAR-10 images.

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
    model = cnn.simple_model.SimpleModel(batch_size, 10)
    builder = cnn.builder.CNNBuilder(images, 3, True, data_format='NHWC')
    return model.inference(builder)


def loss(logits, labels):
    """Calculates total loss including cross-entropy and weight decay terms.

    Args:
        logits: 2D Tensor [BATCH_SIZE, 10]. Logits calculated by inference().
        labels: 1D Tensor [BATCH_SIZE]. Single label of correct class for
        each image

    Returns:
        Loss for the current model.
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    return cross_entropy_mean
