"""
Introduction to CNNs with Tensorflow. Multiple layers of convolution/pooling followed by multiple fully connected
layers.

Achieves approximately 92.5% accuracy.

Will Long
June 8, 2017
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Using an interactive session makes TensorFlow more flexible about how you # structure your code. You can interleave
# operations which build a computation # graph with ones that run the graph. Otherwise we'd have to build the entire
#  graph first and then run it.
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# Initialize weights to be slightly randomized to break symmetries.
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


# Slightly positive initial biases as well as we are using ReLUs and want to avoid dead units.
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# Basic convolution with a stride of 1 and padding to ensure output size = input size.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 2x2 max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# First layer computes 32 features for each 5x5 block in the image.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Reshape input to 4D tensor for proper format. Last dimension is 1 because only 1 input channel.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolve the image with the weight tensor, add the bias, apply ReLU and then max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# In second layer, we will have 64 features for each 5x5 patch. Second pooling reduces image size to 7x7
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Add a fully-connected layer with 1024 neurons to process the entire image. Reshape the tensor from the pooling layer
# into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Apply dropout to the fully connected layer before the readout layer to reduce overfitting. TensorFlow automatically
# scales outputs so we don't need to worry about it. Make keep_prob a placeholder so we can adjust it (i.e. for training
# vs. for testing).
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Add a final readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Training similar to basic model, with cross entropy between labels and softmax activation of readout, but we use a
# more sophisticated optimizer (ADAM), include keep_prob as a parameter, and add logging to every 100th iteration of the
#  training process.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
# 20000 epochs with 50 images per epoch
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        print("Step {}: Training Accuracy = {}%".format(i, train_accuracy * 100.0))
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

# Only test 100 of the test images at a time to avoid crashing due to memory limitations.
accuracies = []
for i in range(0, len(mnist.test.labels), 100):
    accuracies.append(
        accuracy.eval(feed_dict={x: mnist.test.images[i:i + 100], y: mnist.test.labels[i:i + 100], keep_prob: 1.0}))
print("Test Accuracy: {}%".format(np.mean(accuracies) * 100.0))
