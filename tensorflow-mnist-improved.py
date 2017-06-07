"""
Simple improvement upon basic softmax regression model through the addition of
a hidden layer to the neural network.

Achieves approximately 92% accuracy.

Will Long
June 7, 2017
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W_1 = tf.Variable(tf.random_normal([784, 40]))
b_1 = tf.Variable(tf.random_normal([40]))
z_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([40, 10]))
b_2 = tf.Variable(tf.random_normal([10]))
z_2 = tf.matmul(z_1, W_2) + b_2

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=z_2)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(z_2, 1))

# Cast to floats and find mean to get overall prediction accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, z_2: mnist.test.labels}))
