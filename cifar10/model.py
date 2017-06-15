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


# TODO: Constants??

def inference(images):
    """Builds model to make predictions from images.

    Architecture is as follows:
    conv3x3x64/1 -> conv 3x3x64/1 -> norm -> pool3x3/2 -> conv5x5x128/1 -> norm
        -> pool3x3/2 -> fc512 -> fc256 -> softmax10

    ReLU is applied after each convolutional layer and fully connected layer.

    Choices made for better regularization are as follows. Pooling layers have
    slight overlap (stride of 2x2). Both fully connected layers employ
    drop out with probability of 0.5. And weight decay is applied to  all
    weights in the model.

    Args:
        images: 4D Tensor [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3]. The
                batch of images obtained from get_input_batch().

    Returns: logits: 2D Tensor [BATCH_SIZE, 10]. Prediction probabilities
    for the classification of each image.
    """
    # TODO: Write function


def loss(logits, labels):
    """Calculates total loss including cross-entropy and weight decay terms.

    Args:
        logits: 2D Tensor [BATCH_SIZE, 10]. Logits calculated by inference().
        labels: 1D Tensor [BATCH_SIZE]. Single label of correct class for
        each image

    Returns:
        Loss for the current model.
    """
