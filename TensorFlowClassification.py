import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report


# read the dataset
census = pd.read_csv('census_data.csv')
# print(census.head())

# the column we need to predict here is income_bracket which is <=50k or >=50k: classes shown below
classes = census['income_bracket'].unique()
# print(classes)

# the converter lambda function
converter = lambda x: 1 if x == '>50k' else 0
census['income_bracket'] = census['income_bracket'].apply(converter)
# print(census.head())

x_data = census.drop('income_bracket', axis=1)
y_data = census['income_bracket']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=101)

# the columns are: age, workclass, education, education_num, marital_status, occupation, relationship, race, gender,
# capital_gain, capital_loss, hours_per_week, native_country, income_bracket
gender = tf.feature_column.categorical_column_with_vocabulary_list('gender', ['Male', 'Female'])
occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital_status', hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass', hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship', hash_bucket_size=1000)
race = tf.feature_column.categorical_column_with_vocabulary_list('race', ['White', 'Black'])

age = tf.feature_column.numeric_column('age')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')
education_num = tf.feature_column.numeric_column('education_num')

# creating a list of feature columns
feat_cols = [age, workclass, education, education_num, marital_status, occupation, relationship, gender,
             capital_gain, capital_loss, native_country, hours_per_week, race]

# create input function
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=100, shuffle=True, num_epochs=None)

# create the model
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)
model.train(input_func, steps=10000)

# predictions and accuracy, with batch size same as the x_test input and no shuffle
pred_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=len(x_test), shuffle=False)
predictions = list(model.predict(input_fn=pred_func))
final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

# using sklearn classifier report
print(classification_report(y_test, final_preds))