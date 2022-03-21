#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as col
import pandas as pd
import math

import image_formatter
import callback
import model
import aircrafts
import birds
import cifar
import fish
import plants

cycle = []
angle = 0
for _ in image_formatter.color_spaces:
    cycle.append(col.hsv_to_rgb([angle, 1, 1]))
    phi = (1 + math.sqrt(5))/2
    angle += 1/phi
    angle %= 1
angle = 0
for _ in image_formatter.color_spaces:
    cycle.append(col.hsv_to_rgb([angle, 0.5, 1]))
    phi = (1 + math.sqrt(5))/2
    angle += 1/phi
    angle %= 1
matplotlib.cycler(color=cycle)


def main():
    datasets = {"fish": fish, "birds": birds, "plants": plants,
                "cifar": cifar, "aircrafts": aircrafts}
    for ds in datasets:
        dataset = datasets[ds]
        loss = {}
        accuracy = {}
        val_loss = {}
        val_accuracy = {}
        (orig_train_images, orig_test_images,
         train_labels, test_labels) = dataset.load()
        out_size = max(train_labels.max(), test_labels.max())+1
        for color_space in ["RGB", "HSV"]:
            train_images = image_formatter.convert_images(orig_train_images,
                                                          color_space)
            test_images = image_formatter.convert_images(orig_test_images,
                                                         color_space)

            my_model = model.model(out_size)
            history = my_model.fit(train_images, train_labels, epochs=10,
                               validation_data=(test_images, test_labels))
            plot_label = ds + "_" + color_space
            loss[plot_label] = history.history["loss"]
            val_loss["val_" + plot_label] = history.history["val_loss"]
            accuracy[plot_label] = history.history["accuracy"]
            val_accuracy["val_" + plot_label] = history.history["val_accuracy"]

        loss_df = pd.DataFrame(loss)
        val_loss_df = pd.DataFrame(val_loss)
        accuracy_df = pd.DataFrame(accuracy)
        val_accuracy_df = pd.DataFrame(val_accuracy)

        full_loss_df = pd.concat([loss_df, val_loss_df], axis=1)
        full_loss_df.plot(kind="line")
        plt.savefig(ds + "_loss.png")


if __name__ == "__main__":
    main()
