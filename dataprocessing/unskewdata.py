import cv2
import numpy as np
from numpy.linalg import norm
from PIL import Image
import os
import time

rootDir = os.getcwd()
fileToConvert = rootDir + "/data/testing/emnist_letters_test_converted"
SZ = 28 # size of each digit is SZ x SZ
fileData = []

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

with open(fileToConvert + ".csv", "r") as trainingDataFile:
    print("Converting...")
    for record in trainingDataFile:
        data = record.split(",")
        correctLabel = int(data[0])
        del data[0]

        data = np.asarray(data, dtype=np.uint8).reshape((28,28))
        imgUnskewed = deskew(data)
        imgUnskewed = imgUnskewed.flatten()
        data = imgUnskewed.tolist()
        data.insert(0, correctLabel)

        fileData.append(data)
            # y = 1
            # for x in data:
            #     if y == len(data):
            #         convertedDataFile.write(str(x)+"\n")
            #     else:
            #         convertedDataFile.write(str(x)+",")
            #         y += 1

with open(fileToConvert + "_deskewed.csv", "w") as convertedDataFile:
    for x in range(len(fileData)):
        for y in range(len(fileData[x])):
            if y+1 == len(fileData[x]):
                convertedDataFile.write(str(fileData[x][y]))
            else:
                convertedDataFile.write(str(fileData[x][y])+",")
        if x+1 == len(fileData):
            pass
            #Do nothing
        else:
            convertedDataFile.write("\n")