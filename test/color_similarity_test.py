import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import sys
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

def scale(arr):
    arr = arr.astype(float)
    if len(arr.shape) < 3:
        ptp = float(np.ptp(arr))
        if ptp > 0:
            arr = (arr - arr.min())/ptp
        else:
            arr.fill(1)
    else:
        for i in range(len(arr[0,0,:])):
            ptp = float(np.ptp(arr[:,:,i]))
            if ptp > 0:
                arr[:,:,i] = (arr[:,:,i] - arr[:,:,i].min())/ptp
            else:
                arr[:,:,i].fill(1)
    return arr

def shuffle_gen(n, seed):
    if n == 0:
        return []
    if n == 1:
        return [0]
    else:
        shuffle = shuffle_gen(n - 1, int(seed/n))
        insert_pos = seed % n
        shuffle.insert(n - (seed % n) - 1, n - 1)
        return shuffle

def compare(color_model_1, color_model_2, color_models, comparisons):
    channels = len(color_model_1[0,0,:])
    cm1 = color_models[color_model_1]
    cm2 = color_models[color_model_2]
    min_diff = 0
    for x in range(len(cm1[:,0,0])):
        for y in range(len(cm1[0,:,0])):
            for ch in range(channels):
                min_diff += abs(float(cm1[x,y,ch]) - float(cm2[x,y,ch]))
    for shuffle_no in range(1, math.factorial(channels)):
        shuffle = shuffle_gen(channels, shuffle_no)
        diff = 0
        for x in range(len(cm1[:,0,0])):
            for y in range(len(cm1[0,:,0])):
                for ch in range(3):
                    diff += abs(float(cm1[x,y,ch]) - float(cm2[x,y,shuffle[ch]]))
        if diff < min_diff:
            min_diff = diff
    print("diff(" + color_model_1 + ", " + color_model_2 + ") = " + str(min_diff))
    comparisons["diff(" + color_model_1 + ", " + color_model_2 + ")"] = min_diff

color_convertion_codes = {}
for element in cv2.__dict__:
    if "COLOR_BGR2" in element:
        if cv2.__dict__[element] not in color_convertion_codes.values():
            color_convertion_codes[element.replace("COLOR_BGR2", "")] = cv2.__dict__[element]

BGR = cv2.imread("test.png")

color_models = {"BGR": scale(np.asarray(BGR))}
for ccc in color_convertion_codes:
    color_models[ccc] = scale(np.asarray(cv2.cvtColor(BGR, color_convertion_codes[ccc])))

comparisons = {}
for color_model_1 in color_models:
    for color_model_2 in color_models:
        if color_model_1 < color_model_2:
            cm1 = color_models[color_model_1]
            cm2 = color_models[color_model_2]
            if len(cm1.shape) == 3 and len(cm2.shape) == 3:
                if cm1.shape[2] == 3 and cm2.shape[2] == 3:
                    compare(color_model_1, color_model_2, color_models, comparisons)

pd.set_option('display.max_rows', None)
df = pd.DataFrame(comparisons.items())
print(df.sort_values(by=1))
