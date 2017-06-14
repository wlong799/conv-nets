"""
Introduction to core machine learning concepts and how TensorFlow works, by
creating a simple softmax regression model with no hidden layers to classify
handwritten digits in the MNIST data set.

Achieves approximately 91% accuracy

Walkthrough found here:
https://www.tensorflow.org/get_started/mnist/beginners

Will Long
June 7, 2017
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Read in MNIST data sets (mnist.train, mnist.validation, mnist.test)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Placeholder for input data and labels. 784 because MNIST images are each
# 28x28 pixels. 10 because we use "one-hot" labeling, where index of correct
# digit is set to 1 and all others are set to 0. None indicates that first
# dimension (i.e. # examples) can be of any length
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# Variables for the modifiable weights and biases that our model will determine
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Implements model. Basically, each input (pixel) is connected to each output
# (label representing 0-9). This is why weight matrix is 784x10. It is a fully
# connected layer, followed by softmax activation function, with cross-entropy
# as out loss function to measure how "good" the model is.
y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Desire to minimize the cross entropy using basic gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Launch session and train 1000 epochs of batch sizes of 100
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Use argmax to get predicted/correct labels, and get number of correct matches
# as a list of booleans
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Cast to floats and find mean to get overall prediction accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
