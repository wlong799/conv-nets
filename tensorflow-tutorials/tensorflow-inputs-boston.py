"""
Introduction to importing/cleaning/properly formatting input to be used by TensorFlow with use of a dataset about Boston
housing.

Tutorial: https://www.tensorflow.org/get_started/input_fn

Will Long
June 12, 2017
"""
import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

BOSTON_DIR = "boston_data/"
MODEL_DIR = "tmp/boston_model/"
BOSTON_TRAIN = BOSTON_DIR + "boston_train.csv"
BOSTON_TEST = BOSTON_DIR + "boston_test.csv"
BOSTON_PREDICT = BOSTON_DIR + "boston_predict.csv"

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio", "medv"]
FEATURES = COLUMNS[0:-1]
LABEL = COLUMNS[-1]

training_set = pd.read_csv(BOSTON_TRAIN, skipinitialspace=True, skiprows=1, names=COLUMNS)
test_set = pd.read_csv(BOSTON_TEST, skipinitialspace=True, skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv(BOSTON_PREDICT, skipinitialspace=True, skiprows=1, names=COLUMNS)

feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[10, 10], model_dir=MODEL_DIR)


def input_fn(data_set):
    features = {k: tf.constant(data_set[k].values) for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return features, labels


regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
predictions = list(y)
print("Predictions: {}".format(str(predictions)))
