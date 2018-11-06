import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


housing_data = pd.read_csv('cal_housing_clean.csv')
y_val = housing_data['medianHouseValue']
X_val = housing_data.drop('medianHouseValue', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_val, y_val, test_size=0.33, random_state=101)

# scale the featured cols
min_max_scaler = MinMaxScaler()
X_train = pd.DataFrame(data=min_max_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(data=min_max_scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)

# creating feature columns
housingMedianAge = tf.feature_column.numeric_column('housingMedianAge')
totalRooms = tf.feature_column.numeric_column('totalRooms')
totalBedrooms = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
medianIncome = tf.feature_column.numeric_column('medianIncome')
feat_cols = [housingMedianAge, totalRooms, totalBedrooms, population, households, medianIncome]
# creating the input functions
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

# train the model
model = tf.estimator.DNNRegressor(hidden_units=[6, 6, 6], feature_columns=feat_cols)
model.train(input_fn=input_func, steps=1000)

# get predictions and evaluation
results = model.evaluate(input_fn=input_func)
predictions = model.predict(input_func)
print(list(predictions))
print(results)

