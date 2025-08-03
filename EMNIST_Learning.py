# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:46:51 2023

@author: dominika
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

    
(x_train,y_train),(x_test,y_test) = tfds.as_numpy(tfds.load('emnist/letters', data_dir='./data', split=['train', 'test'], as_supervised=True, batch_size=-1))
x_train = x_train.reshape(88800,28,28)
x_test = x_test.reshape(14800,28,28)
for i in range(88800): x_train[i]=np.fliplr(np.rot90(x_train[i],k=-1))
for i in range(14800): x_test[i]=np.fliplr(np.rot90(x_test[i],k=-1))
x_train,x_test=x_train/255.0,x_test/255.00

y_train = tf.keras.utils.to_categorical(y_train-1,26)
y_test = tf.keras.utils.to_categorical(y_test-1,26)

MonReseau = tf.keras.Sequential()
MonReseau.add(tf.keras.layers.Flatten(input_shape=(28,28)))
MonReseau.add(tf.keras.layers.Dense(units=200,activation='relu'))
MonReseau.add(tf.keras.layers.Dense(units=26,activation='softmax'))

MonReseau.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
MonReseau.fit(x=x_train, y=y_train, batch_size=20, epochs=10, validation_data=(x_test,y_test))
MonReseau.save('MonReseau_new.h5')
