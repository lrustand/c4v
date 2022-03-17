#!/usr/bin/env python
### Taken from tensorflow documentation
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import cv2

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#for images in (train_images, test_images):
#    for i in images:
#        cv2.cvtColor(i, cv2.COLOR_BGR2BAYER_RG)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0



class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

class test():
    def __init__(self):
        self.train_batch = 0
        self.logfile = open("asd.csv", "w")
        pass

    def _implements_train_batch_hooks(self):
        return True
    def _implements_test_batch_hooks(self):
        return True
    def _implements_predict_batch_hooks(self):
        return False

    def set_model(self, a):
        print("Model", a)
    def set_params(self, a):
        print("Params", a)

    def on_train_begin(self, a):
        pass
    def on_train_batch_begin(self, a, b):
        pass
    def on_train_batch_end(self, a, b):
        loss = b["loss"]
        accuracy = b["accuracy"]
        #self.logfile.write(f"{self.train_batch},{loss},{accuracy},,\n")
        #self.train_batch += 1
    def on_train_end(self, a):
        pass

    def on_test_begin(self, a):
        pass
    def on_test_batch_begin(self, a, b):
        pass
    def on_test_batch_end(self, a, b):
        pass
    def on_test_end(self, a):
        pass

    def on_epoch_begin(self, a, b):
        pass
    def on_epoch_end(self, a, b):
        loss = b["loss"]
        accuracy = b["accuracy"]
        val_loss = b["val_loss"]
        val_accuracy = b["val_accuracy"]
        self.logfile.write(f"{a},{loss},{accuracy},{val_loss},{val_accuracy}\n")


asd = test()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels), callbacks=[asd])


