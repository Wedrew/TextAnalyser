import cv2
from sys import argv
import numpy as np
from numpy.linalg import norm
from PIL import Image
import os
import time

SZ = 28 # size of each digit is SZ x SZ
unskewedData = []

def unskewImage(image):
    m = cv2.moments(image)
    if abs(m['mu02']) < 1e-2:
        return image.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    image = cv2.warpAffine(image, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return image

def unskewFileData(rootFile, saveFile):
    with open(rootFile, "r") as trainingDataFile:
        for record in trainingDataFile:
            data = record.split(",")
            correctLabel = int(data[0])
            del data[0]
            #Skew data here
            data = np.asarray(data, dtype=np.uint8).reshape((28,28))
            imageUnskewed = deskew(data)
            imageUnskewed = imageUnskewed.flatten()
            data = imageUnskewed.tolist()
            data.insert(0, correctLabel)
            unskewedData.append(data)

    with open(saveFile, "w") as convertedDataFile:
        for x in range(len(unskewedData)):
            for y in range(len(unskewedData[x])):
                if y+1 == len(unskewedData[x]):
                    convertedDataFile.write(str(unskewedData[x][y]))
                else:
                    convertedDataFile.write(str(unskewedData[x][y])+",")
            if x+1 == len(unskewedData):
                pass
                #Do nothing
            else:
                convertedDataFile.write("\n")

def getopts(argv):
    opts = {}
    while argv:
        if argv[0][0] == "-":
            opts[argv[0]] = argv[1]
        argv = argv[1:]
    return opts

if __name__ == "__main__":
    #Get working directory
    rootDir = os.getcwd()
    myargs = getopts(argv)
    if "-f" in myargs and "-s" in myargs:
        rootFile = myargs["-f"]
        saveFile = myargs["-s"]
        unskewFileData(rootFile, saveFile)
    else:
        print("Incorrect arguments")