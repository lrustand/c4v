import numpy as np
import fish
import birds
import cifar
import plants
import aircrafts

import image_formatter

datasets = {"fish": fish, "birds": birds, "plants": plants,
            "cifar": cifar, "aircrafts": aircrafts}

for ds in datasets:
    train_imgs = datasets[ds].load()[0]

    print("\n\n" + ds)

    color_spaces_average_values = []
    for color_space in image_formatter.color_spaces:
        print(color_space)
        converted_imgs = image_formatter.convert_images(train_imgs, color_space)
        average_color = np.median(converted_imgs, axis=(0, 1, 2)) * 255
        print(average_color)
        color_spaces_average_values.append(average_color)
