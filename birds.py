#!/usr/bin/env python
### Taken from tensorflow documentation
import tensorflow as tf
import numpy as np
import image_formatter

from tensorflow.keras import datasets, layers, models
import cv2, glob
import callback

train_images = []
train_labels = []
test_images = []
test_labels = []

def load(width=32, height=32):
    with open("datasets/birds/class_dict.csv") as f:
        for line in f.readlines():
            bird = line.split(",")[1]
            for file in glob.glob('datasets/birds/train/' + bird + '/*.jpg'):
                img = cv2.imread(file)

                img = cv2.resize(img, (width, height))
                train_images.append(np.asarray(img))
                train_labels.append(int(line.split(",")[0]))
            for file in glob.glob('datasets/birds/test/' + bird + '/*.jpg'):
                img = cv2.imread(file)

                img = cv2.resize(img, (width, height))
                test_images.append(np.asarray(img))
                test_labels.append(int(line.split(",")[0]))

    return np.array(train_images), np.array(test_images), np.array(train_labels), np.array(test_labels)

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

