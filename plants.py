#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import image_formatter

from tensorflow.keras import layers, models
import cv2
import glob
from sklearn.model_selection import train_test_split
import callback


def load(width=32, height=32):
    images = []
    labels = []

    label = 0
    with open("datasets/plants/class_names.csv") as f:
        for plant in f.readlines():
            plant = plant.replace(' ', '_').lower().replace('\n', '')
            path = 'datasets/plants/dataset/resized/' + plant
            for file in glob.glob(path + '/*.jpg'):
                img = cv2.imread(file)

                img = cv2.resize(img, (width, height))
                images.append(np.asarray(img))
                labels.append(label)
            label += 1

    images = np.array(images)
    labels = np.array(labels)

    return train_test_split(images, labels, test_size=0.2, random_state=42)


if __name__ == "__main__":
    color_space = "BGR"
    train_images, test_images, train_labels, test_labels = load()
    train_images = image_formatter.convert_images(train_images, color_space)
    test_images = image_formatter.convert_images(test_images, color_space)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(32, 32, 3)))
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
                        validation_data=(test_images, test_labels),
                        callbacks=[asd])
