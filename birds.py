#!/usr/bin/env python
import numpy as np
import pandas as pd
import image_formatter

import cv2
import glob
import model


def load(width=32, height=32):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

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
        birds_model = model.model(out_size)
        history = birds_model.fit(train_images, train_labels, epochs=10,
                                  validation_data=(test_images, test_labels))
        loss["birds_" + color_space] = history.history["loss"]
        val_loss["birds_" + color_space] = history.history["val_loss"]
        accuracy["birds_" + color_space] = history.history["accuracy"]
        val_accuracy["birds_" + color_space] = history.history["val_accuracy"]

    loss_df = pd.DataFrame(loss)
    val_loss_df = pd.DataFrame(val_loss)
    accuracy_df = pd.DataFrame(accuracy)
    val_accuracy_df = pd.DataFrame(val_accuracy)


if __name__ == "__main__":
    main()
