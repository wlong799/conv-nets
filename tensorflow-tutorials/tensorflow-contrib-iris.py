"""
Introduction to the high-level contrib-learn API of TensorFlow using the Iris dataset.

Will Long
June 12, 2017
"""
import os
from urllib.request import urlopen

import tensorflow as tf
import numpy as np

IRIS_DIRECTORY = "iris_data/"

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

if not os.path.exists(IRIS_DIRECTORY):
    os.makedirs(IRIS_DIRECTORY)

if not os.path.exists(IRIS_DIRECTORY + IRIS_TRAINING):
    raw = urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_DIRECTORY + IRIS_TRAINING, 'wb') as f:
        f.write(raw)

if not os.path.exists(IRIS_DIRECTORY + IRIS_TEST):
    raw = urlopen(IRIS_TEST_URL).read()
    with open(IRIS_DIRECTORY + IRIS_TEST, 'wb') as f:
        f.write(raw)

# Load data sets
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_DIRECTORY + IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_DIRECTORY + IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")


# Define the test inputs
def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y


# Fit model.
classifier.fit(input_fn=get_train_inputs, steps=2000)


# Define the test inputs
def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y


# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                     steps=1)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


# Classify two new flower samples.
def new_samples():
    return np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)


predictions = list(classifier.predict(input_fn=new_samples))

print("New Samples, Class Predictions:    {}\n".format(predictions))
