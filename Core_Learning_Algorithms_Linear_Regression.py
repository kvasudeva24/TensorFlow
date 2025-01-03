!pip install tensorflow==2.1.0
!pip install tensorflow-estimator==2.1.0

from __future__ import absolute_import, division, print_function, unicode_literals

# Import tensorflow_estimator after installation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

#pops the columns with certain header
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#shows the first 5 columns
# dftrain.head()

# #describe gives us statistics of the data
# dftrain.describe()

# #(x,y) where x = number of entries, y = 9 different columns
# dftrain.shape

# y_train.head()

# #makes a histogram with bins of size 20
# dftrain.age.hist(bins=20)

# #value_counts() sums up a non quantitative category (bar graph)
# dftrain.sex.value_counts().plot(kind='barh')
# dftrain['class'].value_counts().plot(kind='barh')


#pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

#we can takeaway that the majority of passengers are in their 20s and 30s,
#majority of passengers are male
#majority of passengers were in third class
#females had a much higher chance of surviving


#Before we continue and create/train a model we must convet our categorical data into numeric data.
#We can do this by encoding each category with an integer

#feed feature columns to lin regression model
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique() #gets a list of all unique values from a given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

#load batches of the dataset at once (32,64,128, etc)
#feed batches number of times to our model (epochs)
#if we need to feed our data in batches and multiple times we need an inout function

#now we must create an input function that converts our pandas dataframe into a tf.data.Dataset Object
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000) # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs) # split dataset into batches of 32 and repeat process for number of epochs
    return ds # return a batch of the dataset
  return input_function # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train) # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=False, shuffle=False)


# Use tf.compat.v1.estimator instead of tf.estimator
linear_est = tf_estimator.estimator.LinearClassifier(feature_columns=feature_columns)

#training the model
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn) #get model metrics/stats by testing on data

clear_output()
print(result['accuracy'])

#accuracy is .7386364

result = list(linear_est.predict(eval_input_fn))
#dictionary that represents each prediction
print(dfeval.loc[0])
print(y_eval.loc[0])
print(result[0]['probablities'][1])
