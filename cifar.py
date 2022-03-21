#!/usr/bin/env python
import image_formatter
from tensorflow.keras import datasets

import pandas as pd
import cv2
import callback
import model


def load(width=32, height=32):
    (train_images, train_labels), (test_images, test_labels) = \
        datasets.cifar10.load_data()
    for image in train_images:
        cv2.resize(image, (width, height))
    for image in test_images:
        cv2.resize(image, (width, height))

    return train_images, test_images, train_labels, test_labels


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

        cifar_model = model.model(out_size)
        history = cifar_model.fit(train_images, train_labels, epochs=10,
                                  validation_data=(test_images, test_labels),
                                  callbacks=[asd])
        loss["cifar_" + color_space] = history.history["loss"]
        val_loss["cifar_" + color_space] = history.history["val_loss"]
        accuracy["cifar_" + color_space] = history.history["accuracy"]
        val_accuracy["cifar_" + color_space] = history.history["val_accuracy"]

    loss_df = pd.DataFrame(loss)
    val_loss_df = pd.DataFrame(val_loss)
    accuracy_df = pd.DataFrame(accuracy)
    val_accuracy_df = pd.DataFrame(val_accuracy)


if __name__ == "__main__":
    main()
