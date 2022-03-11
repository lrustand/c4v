#!/usr/bin/env python
### Taken from tensorflow documentation
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras import datasets, layers, models
import cv2, glob
from sklearn.model_selection import train_test_split

color_space = "BGR"

df = pd.read_csv("datasets/fish/final_all_index.txt", sep="=", header=None)
df_cr = pd.read_csv("color_ranges.csv")

df_cr = df_cr.loc[df_cr.iloc[:,0]==color_space,"min":].reset_index()

files = df.iloc[:,3]
all_labels = df.iloc[:,0]
img_types = df.iloc[:,2]

images = []
labels = []

for row in range(len(files)):
    if img_types[row] == "insitu":
        img = cv2.imread("datasets/fish/images/cropped/" + files[row] + ".png")

        img = cv2.resize(img, (32,32))
        if color_space != "BGR":
            img = cv2.cvtColor(img, cv2.__dict__["COLOR_BGR2" + color_space])
        img = np.asarray(img).astype(float)
        for channel in range(img.shape[2]):
            img[:,:,channel] = img[:,:,channel] - df_cr.iloc[channel,0]
            img[:,:,channel] = img[:,:,channel] / df_cr.iloc[channel,2]

        images.append(img)
        labels.append(all_labels[row])

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


