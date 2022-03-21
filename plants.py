#!/usr/bin/env python
import numpy as np
import pandas as pd
import image_formatter

import cv2
import glob
from sklearn.model_selection import train_test_split
import callback
import model


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


def main():
    loss = {}
    accuracy = {}
    val_loss = {}
    val_accuracy = {}
    (orig_train_images, orig_test_images,
     train_labels, test_labels) = load()
    out_size = max(train_labels.max(), test_labels.max())+1
    for color_space in image_formatter.color_spaces:
        train_images = image_formatter.convert_images(orig_train_images,
                                                      color_space)
        test_images = image_formatter.convert_images(orig_test_images,
                                                     color_space)

        asd = callback.test()

        plants_model = model.model(out_size)
        history = plants_model.fit(train_images, train_labels, epochs=10,
                                   validation_data=(test_images, test_labels),
                                   callbacks=[asd])
        loss["plants_" + color_space] = history.history["loss"]
        val_loss["plants_" + color_space] = history.history["val_loss"]
        accuracy["plants_" + color_space] = history.history["accuracy"]
        val_accuracy["plants_" + color_space] = history.history["val_accuracy"]

    loss_df = pd.DataFrame(loss)
    val_loss_df = pd.DataFrame(val_loss)
    accuracy_df = pd.DataFrame(accuracy)
    val_accuracy_df = pd.DataFrame(val_accuracy)


if __name__ == "__main__":
    main()
