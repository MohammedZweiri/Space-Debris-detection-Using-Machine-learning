# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 08:13:55 2021

@author: Mohammed Zweiri (2021) and Iftekher Mamun (2019)
"""


import pandas as pd
import numpy as np
import tensorflow
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
load_img)
from tensorflow.keras.models import Sequential
from keras import optimizers
from keras.preprocessing import image
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import math
import datetime
import time

img_width, img_height = 224,224

top_model_weights_path = 'bottleneck_fc_model.h5'

train_data_location = r'Your directory'
validation_data_location = r'Your directory'
test_data_location = r'Your directory'
epochs = 10
batch_size = 50

vgg16 = applications.VGG16(include_top = False, weights = 'imagenet')

datagen = ImageDataGenerator(rescale = 1. /255)
#Train data
start = datetime.datetime.now()

generator = datagen.flow_from_directory(
    train_data_location,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = None,
    shuffle = False)

nb_train_samples = len(generator.filenames)
num_classes = len(generator.class_indices)

predict_size_train = int(math.ceil(nb_train_samples/batch_size))

bottleneck_features_train = vgg16.predict_generator(generator, predict_size_train)

np.save('bottleneck_features_train.npy', bottleneck_features_train)
end = datetime.datetime.now()
elapsed = end-start
print('Time: ', elapsed)

#Validation data
start = datetime.datetime.now()

generator = datagen.flow_from_directory(
    validation_data_location,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = None,
    shuffle = False)

nb_validation_samples = len(generator.filenames)
num_classes = len(generator.class_indices)

predict_size_validation = int(math.ceil(nb_validation_samples/batch_size))

bottleneck_features_validation = vgg16.predict_generator(generator, predict_size_validation)

np.save('bottleneck_features_validation.npy', bottleneck_features_validation)
end = datetime.datetime.now()
elapsed = end-start
print('Time: ', elapsed)

#Test data
start = datetime.datetime.now()

generator = datagen.flow_from_directory(
    test_data_location,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = None,
    shuffle = False)

nb_test_samples = len(generator.filenames)
num_classes = len(generator.class_indices)

predict_size_test = int(math.ceil(nb_test_samples/batch_size))

bottleneck_features_test = vgg16.predict_generator(generator, predict_size_test)

np.save('bottleneck_features_test.npy', bottleneck_features_test)
end = datetime.datetime.now()
elapsed = end-start
print('Time: ', elapsed)

#Train data
generator_top = datagen.flow_from_directory(
   train_data_location,
   target_size = (img_width, img_height),
   batch_size = batch_size,
   class_mode = 'categorical',
   shuffle = False)

nb_train_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)

train_data = np.load('bottleneck_features_train.npy')
train_labels = generator_top.classes

train_labels = to_categorical(train_labels,
                             num_classes = num_classes)

#Validation data
generator_top = datagen.flow_from_directory(
   validation_data_location,
   target_size = (img_width, img_height),
   batch_size = batch_size,
   class_mode = 'categorical',
   shuffle = False)

nb_validation_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)

validation_data = np.load('bottleneck_features_validation.npy')
validation_labels = generator_top.classes

validation_labels = to_categorical(validation_labels,
                             num_classes = num_classes)

#Test data
generator_top = datagen.flow_from_directory(
   test_data_location,
   target_size = (img_width, img_height),
   batch_size = batch_size,
   class_mode = 'categorical',
   shuffle = False)

nb_test_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)

test_data = np.load('bottleneck_features_test.npy')
test_labels = generator_top.classes

test_labels = to_categorical(test_labels,
                             num_classes = num_classes)

#CNN 
start = datetime.datetime.now()
model = Sequential()
model.add(Flatten(input_shape = train_data.shape[1:]))
model.add(Dense(100, activation = tensorflow.keras.layers.LeakyReLU(alpha = 0.1)))
model.add(Dropout(0.5))
model.add(Dense(100, activation = tensorflow.keras.layers.LeakyReLU(alpha = 0.1)))
model.add(Dropout(0.5))
model.add(Dense(100, activation = tensorflow.keras.layers.LeakyReLU(alpha = 0.1)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', 
             optimizer = tensorflow.keras.optimizers.RMSprop(lr = 1e-4),
             metrics = ['acc'])
history = model.fit(train_data, train_labels,
                   epochs = 75,
                   batch_size = batch_size,
                   validation_data = (validation_data, validation_labels))

model.save_weights(top_model_weights_path)
(eval_loss, eval_accuracy) = model.evaluate(
    validation_data, validation_labels,
    batch_size = batch_size, verbose = 1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
end = datetime.datetime.now()
elapsed = end - start
print('Time: ', elapsed)]

#Plotting the results

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

#Determines Accuracy result
model.evaluate(test_data, test_labels)

#Determines Precision, Recall and F1-score results
preds = np.round(model.predict(test_data),0)
Space_objects = ['Rocket_Body', 'Satellite','Space_debris']
classification_metrics = metrics.classification_report(test_labels, preds,target_names=Space_objects)
print(classification_metrics)





