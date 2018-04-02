import os
import numpy as np
import matplotlib.pyplot
import cv2
from skimage.morphology import medial_axis, skeletonize, skeletonize_3d
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import numpy
from skimage import data
#Mapping for emnist training data, does not include some letters which may have similar
#lower and upper case for better performance
emnistBalancedMapping = {
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9',
    10:'A',
    11:'B',
    12:'C',
    13:'D',
    14:'E',
    15:'F',
    16:'G',
    17:'H',
    18:'I',
    19:'J',
    20:'K',
    21:'L',
    22:'M',
    23:'N',
    24:'O',
    25:'P',
    26:'Q',
    27:'R',
    28:'S',
    29:'T',
    30:'U',
    31:'V',
    32:'W',
    33:'X',
    34:'Y',
    35:'Z',
    36:'a',
    37:'b',
    38:'d',
    39:'e',
    40:'f',
    41:'g',
    42:'h',
    43:'n',
    44:'q',
    45:'r',
    46:'t',
}
emnistLetterMapping = {
    0:'A',
    1:'B',
    2:'C',
    3:'D',
    4:'E',
    5:'F',
    6:'G',
    7:'H',
    8:'I',
    9:'J',
    10:'K',
    11:'L',
    12:'M',
    13:'N',
    14:'O',
    15:'P',
    16:'Q',
    17:'R',
    18:'S',
    19:'T',
    20:'U',
    21:'V',
    22:'W',
    23:'X',
    24:'Y',
    25:'Z',
}

fileName = str(input("File name: "))
filePath = "/Users/andrewpagan/Documents/School/TextAnalyser/data/testing/" + fileName

with open(filePath, "r") as file:
    x=1
    for record in file:
        data = record.split(",")
        correctLetter = emnistLetterMapping[int(data[0])]
        print("Correct letter: {}".format(correctLetter))
        del data[0]
        data = np.asarray(data, np.uint8)


        imageArray= np.asarray(data).reshape((28,28))
        # imageArray = 255-imageArray
        # imageArray = cv2.resize(imageArray, None, fx=5, fy=5, interpolation = cv2.INTER_CUBIC)
        cv2.imshow("asdf", imageArray)
        cv2.waitKey(0)