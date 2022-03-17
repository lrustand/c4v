#!/usr/bin/env python
### Taken from tensorflow documentation
import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, layers, models
import cv2, glob
import callback

train_images = []
train_labels = []
test_images = []
test_labels = []

with open("datasets/birds/class_dict.csv") as f:
    for line in f.readlines():
        bird = line.split(",")[1]
        for file in glob.glob('datasets/birds/train/' + bird + '/*.jpg'):
            img = cv2.imread(file)

            img = cv2.resize(img, (32,32))
            #cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            train_images.append(np.asarray(img) / 255.0)
            train_labels.append(int(line.split(",")[0]))
        for file in glob.glob('datasets/birds/test/' + bird + '/*.jpg'):
            img = cv2.imread(file)

            img = cv2.resize(img, (32,32))
            #cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            test_images.append(np.asarray(img) / 255.0)
            test_labels.append(int(line.split(",")[0]))

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


