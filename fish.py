#!/usr/bin/env python
### Taken from tensorflow documentation
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras import datasets, layers, models
import cv2, glob
from sklearn.model_selection import train_test_split
import image_formatter
import callback

def load(width=32, height=32):
    df = pd.read_csv("datasets/fish/final_all_index.txt", sep="=", header=None)

    files = df.iloc[:,3]
    all_labels = df.iloc[:,0]
    img_types = df.iloc[:,2]

    images = []
    labels = []

    for row, file in enumerate(files):
        if img_types[row] == "insitu":
            img = cv2.imread("datasets/fish/images/cropped/" + file + ".png")
            img = cv2.resize(img, (width, height))
            images.append(img)
            labels.append(all_labels[row])

    images = np.array(images)
    labels = np.array(labels)
    return train_test_split(images, labels, test_size=0.2, random_state=42)


if __name__ == "__main__":
    color_space = "BGR"
    train_images, test_images, train_labels, test_labels = load()
    train_images = image_formatter.convert_images(train_images, color_space)
    test_images = image_formatter.convert_images(test_images, color_space)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(max(train_labels.max(), test_labels.max())+1))

    model.summary()

    asd = callback.test()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=100,
                        validation_data=(test_images, test_labels), callbacks=[asd])

