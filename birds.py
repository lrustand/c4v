#!/usr/bin/env python
### Taken from tensorflow documentation
import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, layers, models
import cv2, glob
from sklearn.model_selection import train_test_split

train_images = []
train_labels = []
test_images = []
test_labels = []

with open("datasets/birds/class_dict.csv") as f:
    for line in f.readlines():
        bird = line.split(",")[1]
        for file in glob.glob('datasets/birds/train' + bird + '/*.jpg'):
            img = cv2.imread(file)

            img = cv2.resize(img, (32,32))
            #cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            train_images.append(np.asarray(img) / 255.0)
            train_labels.append(int(line.split(",")[0]))
        for file in glob.glob('datasets/birds/test' + bird + '/*.jpg'):
            img = cv2.imread(file)

            img = cv2.resize(img, (32,32))
            #cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            test_images.append(np.asarray(img) / 255.0)
            test_labels.append(int(line.split(",")[0]))
        label += 1
        label += 1

train_images = np.array(images)
train_labels = np.array(labels)
test_images = np.array(images)
test_labels = np.array(labels)


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

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100,
                    validation_data=(test_images, test_labels))


