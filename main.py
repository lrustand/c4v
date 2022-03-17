#!/usr/bin/env python

import tensorflow as tf
import image_formatter
import callback
import model
import fish
from tensorflow.keras import layers, models




if __name__ == "__main__":
    color_space = "BGR"
    train_images, test_images, train_labels, test_labels = fish.load()
    train_images = image_formatter.convert_images(train_images, color_space)
    test_images = image_formatter.convert_images(test_images, color_space)

    out_size = max(train_labels.max(), test_labels.max())+1
    fish_model = model.model(out_size)



    history = model.fit(train_images, train_labels, epochs=100,
                        validation_data=(test_images, test_labels), callbacks=[asd])
