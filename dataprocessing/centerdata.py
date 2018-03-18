from PIL import Image, ImageFilter, ImageDraw, ImageOps
import PIL.ImageOps
from scipy import ndimage
from scipy import misc
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from pylab import *

#Centers and scales image
def centerData(imageArray):
    #Create image
    im = Image.fromarray(imageArray)
    pix = imageArray
    pix = pix[:,:] # Drop the alpha channel
    idx = np.where(pix-255)[0:2] # Drop the color when finding edges
    box = list(map(min,idx))[::] + list(map(max,idx))[::]
    #Crop and apply as numpy array
    region = im.crop(box)
    region_pix = np.asarray(region)

    img = Image.fromarray(region_pix)
    pixels = list(img.getdata())
    width, height = img.size

    if width == height:
        ratio = int(width/10)
        im2 = Image.new("L", (width, height), 255)
        im2.paste(img, (0, 0))
        img2 = ImageOps.expand(im2, border=ratio,fill='black')
        img2 = img2.convert("L")
        img2 = PIL.ImageOps.invert(img2)
        img2 = img2.resize((28,28))
        img2.show()
        time.sleep(1)
        array = np.asarray(img2)
        return array.flatten()
    elif width > height:
        #Will add one tenth the squares value as padding to longest side
        ratio = int(width/10)
        im2 = Image.new("L", (width, width), 255)
        im2.paste(img, (0, int(width/2)-int(height/2)))
        img2 = ImageOps.expand(im2, border=ratio,fill='black')
        img2 = img2.convert("L")
        img2 = PIL.ImageOps.invert(img2)
        img2 = img2.resize((28,28))
        array = np.asarray(img2)
        return array.flatten()
    elif height > width:
        #Will add one tenth the squares value as padding to longest side
        ratio = int(height/10)
        im2 = Image.new("L", (height, height), 255)
        im2.paste(img, (int(height/2)-int(width/2), 0))
        img2 = ImageOps.expand(im2, border=ratio,fill='black')
        img2 = img2.convert("L")
        img2 = PIL.ImageOps.invert(img2)
        img2 = img2.resize((28,28))
        array = np.asarray(img2)
        return array.flatten()

    # subplot(121)
    # imshow(pix)
    # subplot(122)
    # imshow(region_pix)
    # show()

#Emnist conversion script that takes emnist data rotates it 90 degrees left, and flips the horizontal axis.
#Not sure why they provided the training data this way
rootDir = os.getcwd()
fileToConvert = rootDir + "/data/testing/emnist_letters_test"

with open(fileToConvert + "_converted.csv", "a") as convertedDataFile:
    with open(fileToConvert + ".csv", "r") as trainingDataFile:
        for record in trainingDataFile:
            data = record.split(",")
            correctLabel = int(data[0])
            del data[0]

            data = [int(x) for x in data]
            data = np.asarray(data, dtype=np.int16)
            data = np.resize(data, (28,28))
            data = centerData(data)
            data = data.tolist()
            
            data.insert(0, correctLabel)

            y = 1
            for x in data:
                if y == len(data):
                    convertedDataFile.write(str(x))
                else:
                    convertedDataFile.write(str(x)+",")
                    y += 1
            convertedDataFile.write('\n')