#!/usr/bin/env python
import numpy as np
import pandas as pd

import cv2
from sklearn.model_selection import train_test_split
import image_formatter
import callback
import model


def load(width=32, height=32):
    df = pd.read_csv("datasets/fish/final_all_index.txt", sep="=", header=None)

    files = df.iloc[:, 3]
    all_labels = df.iloc[:, 0]
    img_types = df.iloc[:, 2]

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

        fish_model = model.model(out_size)
        history = fish_model.fit(train_images, train_labels, epochs=10,
                                 validation_data=(test_images, test_labels),
                                 callbacks=[asd])
        loss["fish_" + color_space] = history.history["loss"]
        val_loss["fish_" + color_space] = history.history["val_loss"]
        accuracy["fish_" + color_space] = history.history["accuracy"]
        val_accuracy["fish_" + color_space] = history.history["val_accuracy"]

    loss_df = pd.DataFrame(loss)
    val_loss_df = pd.DataFrame(val_loss)
    accuracy_df = pd.DataFrame(accuracy)
    val_accuracy_df = pd.DataFrame(val_accuracy)


if __name__ == "__main__":
    main()
