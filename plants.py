#!/usr/bin/env python
### Taken from tensorflow documentation
import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, layers, models
import cv2, glob
from sklearn.model_selection import train_test_split

images = []
labels = []

label = 0
with open("datasets/plants/class_names.csv") as f:
    for plant in f.readlines():
        plant = plant.replace(' ', '_').lower().replace('\n','')
        for file in glob.glob('datasets/plants/dataset/resized/' + plant + '/*.jpg'):
            img = cv2.imread(file)

            img = cv2.resize(img, (32,32))
            #cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            images.append(np.asarray(img) / 255.0)
            labels.append(label)
        label += 1

images = np.array(images)
labels = np.array(labels)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(labels.max()+1))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100,
                    validation_data=(test_images, test_labels))


