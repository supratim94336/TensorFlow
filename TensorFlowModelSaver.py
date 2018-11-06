import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline

# creating data
np.random.seed(101)
x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

# plotting to understand how the data looks like
plt.plot(x_data, y_label, '*')

# creating slope and intercept variable
tf.set_random_seed(101)
m = tf.Variable(0.39)
b = tf.Variable(0.2)

# cost function
error = tf.reduce_mean(y_label - (m * x_data + b))

# init and saver model
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
# ----------------------------------------------------------------------------------------------------------------------
# Saving a Model
# ----------------------------------------------------------------------------------------------------------------------
with tf.Session() as sess:
    sess.run(init)

    epochs = 100

    for i in range(epochs):
        sess.run(train)

    # Fetch Back Results
    final_slope, final_intercept = sess.run([m, b])

    saver.save(sess, 'new_models/my_second_model.ckpt')

# evaluate results
x_test = np.linspace(-1, 11, 10)
y_pred_plot = final_slope * x_test + final_intercept
plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')

# ----------------------------------------------------------------------------------------------------------------------
# Loading a Model
# ----------------------------------------------------------------------------------------------------------------------
with tf.Session() as sess:
    # Restore the model
    saver.restore(sess, 'new_models/my_second_model.ckpt')

    # Fetch Back Results
    restored_slope, restored_intercept = sess.run([m, b])

x_test = np.linspace(-1, 11, 10)
y_pred_plot = restored_slope*x_test + restored_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')