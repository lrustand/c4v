import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import sys

def test(arr, string):
    print(string + str(arr.shape))
    print(string + ": " + str(arr.min()) + " - " + str(arr.max()))
    print()

def test_nD(arr, string):
    print(string + str(arr.shape))
    for i in range(len(arr[0,0,:])):
        print(string + " " + str(i) + ": " + str(arr[:,:,i].min()) + " - " + str(arr[:,:,i].max()))
    print()

def scale(arr):
    arr = arr.astype(float)
    if len(arr.shape) < 3:
        arr = (arr - arr.min())/(arr.max() - arr.min())
    else:
        for i in range(len(arr[0,0,:])):
            arr[:,:,i] = (arr[:,:,i] - arr[:,:,i].min())/(arr[:,:,i].max() - arr[:,:,i].min())
            return arr

def compare(color_model_1, color_model_2, color_models):
    cm1 = color_models[color_model_1]
    cm2 = color_models[color_model_2]
    min_diff = 0
    for x in range(len(cm1[:,0,0])):
        for y in range(len(cm1[0,:,0])):
            for ch in range(len(cm1[0,0,:])):
                min_diff += abs(float(cm1[x,y,ch]) - float(cm2[x,y,ch]))
    comparrisons["diff(" + color_model_1 + ", " + color_model_2 + ")"] = min_diff

color_convertion_codes = {}
for element in cv2.__dict__:
    if "COLOR_BGR2" in element:
        color_convertion_codes[element.replace("COLOR_BGR2", "")] = cv2.__dict__[element]

BGR = cv2.imread("test.png")
color_models = {}

for ccc in color_convertion_codes:
    color_models[ccc] = scale(np.asarray(cv2.cvtColor(BGR, color_convertion_codes[ccc])))
color_models["BGR"] = scale(np.asarray(BGR))
print(color_models.keys())

color_models = {"YUV": color_models["YUV"],
                "RGB": color_models["RGB"]}

comparrisons = {}
for color_model_1 in color_models:
    for color_model_2 in color_models:
        if color_model_1 < color_model_2:
            compare(color_model_1, color_model_2, color_models, comparrisons)
