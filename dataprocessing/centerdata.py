import numpy as np
from sys import argv
import cv2
import os
import time

#Centers and scales image
def centerImage(imageArray, threshold):
    THRESHOLD = threshold
    borderType = cv2.BORDER_CONSTANT
    #Get bordered area where values are 0
    points = np.argwhere(imageArray>=THRESHOLD)
    points = np.fliplr(points)
    x, y, w, h = cv2.boundingRect(points)
    finalImage = imageArray[y:y+h, x:x+w]
    height, width = finalImage.shape
    # # Adjusts for rounding errors when cropped image had an odd side length
    # while height < width:
    #     finalImage= cv2.copyMakeBorder(finalImage, 1, 1, 0, 0, borderType, None, 0)
    #     height = finalImage.shape[0]
    # while width < height:
    #     finalImage= cv2.copyMakeBorder(finalImage, 0, 0, 1, 1, borderType, None, 0)
    #     width = finalImage.shape[1]
    #We have a centered image using WIDTHxHEIGHT
    return finalImage

#Centers and scales image
def centerFileData(rootFile, saveFile):
    HEIGHT = int(input("Height: "))
    WIDTH = int(input("Width: "))
    THRESHOLD = int(input("THRESHOLD: "))
    centeredData = []
    borderType = cv2.BORDER_CONSTANT
    with open(rootFile, "r") as dataFile:
        for record in dataFile:
            #Split file
            allValues = record.split(",")
            correctLabel = allValues[0]
            inputs = (np.asarray(allValues[1:], dtype=np.uint8))
            imageArray = inputs.reshape((HEIGHT,WIDTH))
            height, width = imageArray.shape
            #Get bordered area where values are 0
            points = np.argwhere(imageArray>=THRESHOLD)
            points = np.fliplr(points)
            x, y, w, h = cv2.boundingRect(points)
            crop = imageArray[y:y+h, x:x+w]
            height, width = crop.shape
            #Find necessary ratios
            top = int((HEIGHT-height)/2)
            left = int((WIDTH-width)/2)
            bottom = top
            right = left
            #Add border
            finalImage= cv2.copyMakeBorder(crop, top, bottom, left, right, borderType, None, 0)
            height, width = finalImage.shape
            # Adjusts for rounding errors when cropped image had an odd side length
            while height < HEIGHT:
                finalImage= cv2.copyMakeBorder(finalImage, 1, 0, 0, 0, borderType, None, 0)
                height = finalImage.shape[0]

            while width < WIDTH:
                finalImage= cv2.copyMakeBorder(finalImage, 0, 0, 1, 0, borderType, None, 0)
                width = finalImage.shape[1]
            #We have a centered image using WIDTHxHEIGHT
            finalImage = finalImage.flatten()
            finalImage = finalImage.tolist()
            #Insert correct label
            finalImage.insert(0, correctLabel)
            centeredData.append(finalImage)

    with open(saveFile, "w") as convertedDataFile:
        for x in range(len(centeredData)):
            for y in range(len(centeredData[x])):
                if y+1 == len(centeredData[x]):
                    convertedDataFile.write(str(centeredData[x][y]))
                else:
                    convertedDataFile.write(str(centeredData[x][y])+",")
            if x+1 == len(centeredData):
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
        centerFileData(rootFile, saveFile)
    else:
        print("Incorrect arguments")