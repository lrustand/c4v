#!/usr/bin/env python
import numpy as np
import pandas as pd
import image_formatter

import cv2
import callback
import model


def load(width=32, height=32):
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
            img_name = line.split()[0] + ".jpg"
            manufacturer = line[len(img_name)-3:]
            img = cv2.imread("datasets/aircrafts/data/images/" + img_name)
            img = cv2.resize(img, (width, height))

            train_images.append(np.asarray(img))
            train_labels.append(manufacturers[manufacturer])

    with open("datasets/aircrafts/data/images_manufacturer_test.txt") as f:
        for line in f.read().splitlines():
            img_name = line.split()[0] + ".jpg"
            manufacturer = line[len(img_name)-3:]
            img = cv2.imread("datasets/aircrafts/data/images/" + img_name)
            img = cv2.resize(img, (width, height))

            test_images.append(np.asarray(img))
            test_labels.append(manufacturers[manufacturer])

    return (np.array(train_images), np.array(test_images),
            np.array(train_labels), np.array(test_labels))


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

        aircrafts_model = model.model(out_size)
        history = aircrafts_model.fit(train_images, train_labels, epochs=10,
                                      validation_data=(test_images, test_labels),
                                      callbacks=[asd])
        loss["aircrafts_" + color_space] = history.history["loss"]
        val_loss["aircrafts_" + color_space] = history.history["val_loss"]
        accuracy["aircrafts_" + color_space] = history.history["accuracy"]
        val_accuracy["aircrafts_" + color_space] = history.history["val_accuracy"]

    loss_df = pd.DataFrame(loss)
    val_loss_df = pd.DataFrame(val_loss)
    accuracy_df = pd.DataFrame(accuracy)
    val_accuracy_df = pd.DataFrame(val_accuracy)


if __name__ == "__main__":
    main()
