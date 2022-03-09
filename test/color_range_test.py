import numpy as np
import cv2
import pandas as pd
import math

def test(arr, string):
    print(string + str(arr.shape))
    print(string + ": " + str(arr.min()) + " - " + str(arr.max()))
    print()

def test_nD(arr, string):
    print(string + str(arr.shape))
    for i in range(len(arr[0,0,:])):
        print(string + " " + str(i) + ": " + str(arr[:,:,i].min()) + " - " + str(arr[:,:,i].max()))
    print()

color_conversion_codes = {}
for element in cv2.__dict__:
    if "COLOR_BGR2" in element:
        if cv2.__dict__[element] not in color_conversion_codes.values():
            color_conversion_codes[element.replace("COLOR_BGR2", "")] = cv2.__dict__[element]

BGR = cv2.imread("test.png")

color_models = {"BGR": np.asarray(BGR)}
for ccc in color_conversion_codes:
    color_models[ccc] = np.asarray(cv2.cvtColor(BGR, color_conversion_codes[ccc]))

for cm in color_models:
    if(len(color_models[cm].shape) == 2):
        test(color_models[cm], cm)
    elif(len(color_models[cm].shape) == 3):
        test_nD(color_models[cm], cm)
