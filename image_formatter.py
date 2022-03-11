#!/usr/bin/env python
import pandas as pd
import numpy as np
import cv2

df = pd.read_csv("color_ranges.csv")

color_ranges = {}
for _, row in df.iterrows():
    if row["format"] not in color_ranges:
        color_ranges[row["format"]] = []
    color_ranges[row["format"]].append((row["min"],row["range"]))

def convert_image(image, format="BGR", width=32, height=32):
    image = cv2.resize(image, (width, height))
    if format == "HLS":
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL))
    elif format == "HSV":
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL))
    elif format == "Lab":
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2Lab))
    elif format == "Luv":
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2Luv))
    elif format == "RGB":
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif format == "XYZ":
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2XYZ))
    elif format == "YCrCb":
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))
    elif format == "YUV":
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2YUV))
    elif format != "BGR":
        raise ValueError("Invalid format selected: " + format)
    image = image.astype(float)
    for channel in range(image.shape[2]):
        image[:,:,channel] = (image[:,:,channel] - color_ranges[format][channel][0])/color_ranges[format][channel][1]
    return image

