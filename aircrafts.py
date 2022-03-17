#!/usr/bin/env python
### Taken from tensorflow documentation
import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, layers, models
import cv2, glob
from sklearn.model_selection import train_test_split
import callback

manufacturers = {}
with open("datasets/aircrafts/data/manufacturers.txt") as f:
    n = 0
    for manufacturer in f.read().splitlines():
        manufacturers[manufacturer] = n
        n += 1

train_images = []
train_labels = []
test_images = []
test_labels = []

with open("datasets/aircrafts/data/images_manufacturer_train.txt") as f:
    for line in f.read().splitlines():
        img_name = line.split()[0]
        manufacturer = line[len(img_name)+1:]
        img = cv2.imread("datasets/aircrafts/data/images/" + img_name + ".jpg")

        img = cv2.resize(img, (32,32))
        #cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        train_images.append(np.asarray(img) / 255.0)
        train_labels.append(manufacturers[manufacturer])

with open("datasets/aircrafts/data/images_manufacturer_test.txt") as f:
    for line in f.read().splitlines():
        img_name = line.split()[0]
        manufacturer = line[len(img_name)+1:]
        img = cv2.imread("datasets/aircrafts/data/images/" + img_name + ".jpg")

        img = cv2.resize(img, (32,32))
        #cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        test_images.append(np.asarray(img) / 255.0)
        test_labels.append(manufacturers[manufacturer])

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(max(test_labels.max(), train_labels.max())+1))

model.summary()

asd = callback.test()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100,
                    validation_data=(test_images, test_labels), callbacks=[asd])


