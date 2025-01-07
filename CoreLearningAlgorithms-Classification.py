from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd

#classification is used to to separate data points into classes of different labels

'''
3 species of flowers: Setosa, Versicolor, Virginica

Information about each flower
    Sepal length
    sepal width
    petal length
    petal width

'''

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train.head()


train_y = train.pop('Species')
test_y = test.pop('Species')

train.head()

def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset  = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#Building the model
#Either a DNN or a Linear Classifier
#DNN > Linear Classifier because there may not be a linear correspondence

classifier = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns
    #two hidden layers of 30 and 10 nodes
    hidden_units[30,10]
    #model must choose between 3 classes
    n_classes = 3
)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000)

# a lambda is an anonymous function in one line

classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))

#test accuracy is 80%

#now we make a prediction

def predicton_fn(features, batch_size = 256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False
    
predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions: 
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probablilities'][0]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability)) 

