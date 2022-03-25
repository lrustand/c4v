#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as col
import pandas as pd
import math
from pathlib import Path

import image_formatter
import model
import aircrafts
import birds
import cifar
import fish
import plants

#cycle = []
#angle = 0
#for _ in image_formatter.color_spaces:
#    cycle.append(col.hsv_to_rgb([angle, 1, 1]))
#    phi = (1 + math.sqrt(5))/2
#    angle += 1/phi
#    angle %= 1
#angle = 0
#for _ in image_formatter.color_spaces:
#    cycle.append(col.hsv_to_rgb([angle, 0.5, 1]))
#    phi = (1 + math.sqrt(5))/2
#    angle += 1/phi
#    angle %= 1


def main():
    models={"small":model.mini_model, "normal": model.model}
    datasets = {"fish": fish, "birds": birds, "plants": plants,
                "cifar": cifar, "aircrafts": aircrafts}
    for current_model in models:
        for run in range(5):
            for ds in datasets:
                dataset = datasets[ds]
                loss = {}
                accuracy = {}
                val_loss = {}
                val_accuracy = {}
                (orig_train_images, orig_test_images,
                 train_labels, test_labels) = dataset.load()
                out_size = max(train_labels.max(), test_labels.max())+1
                for color_space in image_formatter.color_spaces:
                    train_images = image_formatter.convert_images(orig_train_images,
                                                                  color_space)
                    test_images = image_formatter.convert_images(orig_test_images,
                                                                 color_space)

                    my_model=models[current_model](out_size)
                    history = my_model.fit(train_images, train_labels, epochs=25,
                                       validation_data=(test_images, test_labels))
                    plot_label = ds + "_" + color_space
                    loss[plot_label] = history.history["loss"]
                    val_loss[plot_label] = history.history["val_loss"]
                    accuracy[plot_label] = history.history["accuracy"]
                    val_accuracy[plot_label] = history.history["val_accuracy"]

                    loss_df = pd.DataFrame(loss)
                    val_loss_df = pd.DataFrame(val_loss)
                    accuracy_df = pd.DataFrame(accuracy)
                    val_accuracy_df = pd.DataFrame(val_accuracy)

                    common_path = f"{run+1}/{current_model}_{ds}."
                    val_acc_path = f"runs/val/acc/{common_path}"
                    val_loss_path = f"runs/val/loss/{common_path}"
                    test_acc_path = f"runs/test/acc/{common_path}"
                    test_loss_path = f"runs/test/loss/{common_path}"

                    filepath = Path(val_acc_path)
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath = Path(val_loss_path)
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath = Path(test_acc_path)
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    filepath = Path(test_loss_path)
                    filepath.parent.mkdir(parents=True, exist_ok=True)

                    loss_df.plot(kind="line")
                    plt.savefig(f"{test_loss_path}png")
                    loss_df.to_csv(f"{test_loss_path}csv")
                    plt.close("all")

                    val_loss_df.plot(kind="line")
                    plt.savefig(f"{val_loss_path}png")
                    val_loss_df.to_csv(f"{val_loss_path}csv")
                    plt.close("all")

                    accuracy_df.plot(kind="line")
                    plt.savefig(f"{test_acc_path}png")
                    accuracy_df.to_csv(f"{test_acc_path}csv")
                    plt.close("all")

                    val_accuracy_df.plot(kind="line")
                    plt.savefig(f"{val_acc_path}png")
                    val_accuracy_df.to_csv(f"{val_acc_path}csv")
                    plt.close("all")


if __name__ == "__main__":
    main()
