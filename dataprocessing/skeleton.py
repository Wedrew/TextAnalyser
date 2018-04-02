import cv2
import numpy as np
import os
from sys import argv
from skimage import img_as_bool, io, color, morphology

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

def skeletonImage(imageArray, dilate, scale):
	imageArray = img_as_bool(imageArray)
	out = morphology.skeletonize(imageArray)
	out = out.astype(np.uint8)*255
	kernel = np.ones((dilate,dilate), np.uint8)
	out = cv2.dilate(out, kernel, iterations=1)
	out = cv2.resize(out, (scale,scale), cv2.INTER_AREA)
	return out

def skeletonFileData(rootFile, saveFile):
	skeletonImages = []
	with open(rootFile, "r") as dataFile:
		x=1
		for record in dataFile:
			data = record.split(",")
			correctLetter = int(data[0])
			del data[0]
			data = np.asarray(data, np.uint8)
			img = data.reshape((28,28))
			
			skel = skeletonImage(img, 5, 28)
			 
			skel = cv2.dilate(skel, kernel, iterations=1)
			skel = skel.flatten()
			skel = skel.tolist()
			skel.insert(0, correctLetter)
			skeletonImages.append(skel)

	with open(saveFile, "w") as convertedDataFile:
		for x in range(len(skeletonImages)):
			for y in range(len(skeletonImages[x])):
				if y+1 == len(skeletonImages[x]):
					convertedDataFile.write(str(skeletonImages[x][y]))
				else:
					convertedDataFile.write(str(skeletonImages[x][y])+",")
			if x+1 == len(skeletonImages):
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
        skeletonFileData(rootFile, saveFile)
    else:
        print("Incorrect arguments")
 