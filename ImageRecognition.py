#Image Recognition using CNN

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
import os
import seaborn as sns
from tensorflow.python.framework import ops
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

#To use GPU for model training
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

training_data = pd.DataFrame(columns=('name','path','label'))


TRAINING_DIR='C:\\Users\\sahil\\Projects\\Cars\\fruits'
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split= .2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale = 1./255,validation_split= .2,)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(64,64),
    color_mode='rgb',
    class_mode='categorical',
    subset='training',
    batch_size=32,
    #shuffle=TRUE,
    #seed=24
)

validation_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(64,64),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    subset='validation',
    #seed=24
)


##Neural Network
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    # The third convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Dropout(0.2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),

    #  neuron hidden layer
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(7, activation='softmax')
])


opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

#Fit Model
history = model.fit(train_generator,
                    epochs=20,
                    #steps_per_epoch=len(train_generator),
                    validation_data = validation_generator,
                    #verbose = 1,
                    #validation_steps=len(validation_generator)
                   )

