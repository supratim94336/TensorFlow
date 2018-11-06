import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# print("Basic dataset for starting up")
# x_data = np.linspace(0.0, 10.0, 1000000)
# noise = np.random.randn(len(x_data))
# y_true = (0.5 * x_data) + 5 + noise
# x_df = pd.DataFrame(data=x_data, columns=['x_data'])
# y_df = pd.DataFrame(data=y_true, columns=['y'])
# my_data = pd.concat([x_df, y_df], axis=1)
# ----------------------------------------------------------------------------------------------------------------------
# print("Estimator Basics")
# feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
# estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)
# x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)
#
# input_func = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y=y_train, batch_size=8, num_epochs=None,
#                                                 shuffle=True)
# train_input_func = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y=y_train, batch_size=8, num_epochs=1000,
#                                                       shuffle=False)
# eval_input_func = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y=y_eval, batch_size=8, num_epochs=1000,
#                                                      shuffle=False)
# print("---------------------------------------> Training")
# estimator.train(input_fn=input_func, steps=1000)
# print("---------------------------------------> Evaluation")
# train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
# print("---------------------------------------")
# eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)
# print('TRAINING DATA METRICS')
# print(train_metrics)
# print('TRAINING EVAL METRICS')
# print(eval_metrics)
#
# brand_new_data = np.linspace(0, 10, 10)
# input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': brand_new_data}, shuffle=False)
# # predictions = list(estimator.predict(input_fn=input_fn_predict))
# #
# # print(predictions)
#
# predictions = []
#
# for pred in estimator.predict(input_fn=input_fn_predict):
#     predictions.append(pred['predictions'])
#
# my_data.sample(n=250).plot(kind='scatter', x='x_data', y='y')
# plt.plot(brand_new_data, predictions, 'r')
# plt.show()
# ----------------------------------------------------------------------------------------------------------------------
diabetes = pd.read_csv('pima-indians-diabetes.csv')
# print(diabetes.columns)
# print(diabetes.head())
# age isn't normalized because it should be used as it is and later we can convert it into age buckets for converting it
# into a categorical grouping column
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps', 'Insulin', 'BMI', 'Pedigree']
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min())/(x.max() - x.min()))
# print(diabetes.head())
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')
# When you now the groups and you also know that they are few
# assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])
# Here 10 is a maximum number in order to say that there can be at max 10 categories
assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

# Feature engineering: visualize features before start learning from it
# diabetes['Age'].hist(bins=20)
# plt.show()

# Making distribution of continuous values into buckets of ranges
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])

# finally all the columns we need for classification
# feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, assigned_group, age_bucket]

# train-test split
x_data = diabetes.drop('Class', axis=1)
labels = diabetes['Class']
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, random_state=101)

# Epochs: One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
# Batches: Total number of training examples present in a single batch
# Iterations: Iterations is the number of batches needed to complete one epoch
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000,
                                                 shuffle=True)
# linear classifier ----------------------------------------------------------------------------------------------------
# model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
# model.train(input_fn=input_func, steps=1000)

# DNN classifier (3 layers, 10 hidden units each and everything is connected to everything) ----------------------------
# ValueError: Items of feature_columns must be a _DenseColumn. You can wrap a categorical column with an embedding_
# column or indicator_column.
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embedded_group_col, age_bucket]
model = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2)
model.train(input_fn=input_func, steps=1000)

results = model.evaluate(input_fn=input_func)
predictions = model.predict(input_func)
print(list(predictions))
print(results)


