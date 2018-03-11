from PIL import Image, ImageFilter
import PIL.ImageOps
from sys import argv
import numpy as np
import matplotlib.pyplot
import os
import time
import random
import re

#Utility to find all images within a folder and combine them into one csv file
#Searches every subdirectory of the folder you give
#Example ---> python buildimagedata.py -p PathToYourFolderWithImages -s FolderToSaveSvgTo
def createCsv(rootDir, savePath, scaleSize):
	imageList = []

	#Find all png files in every lower directory and add them to imageList
	for root, dirs, files in os.walk(rootDir):
		for file in files:
			if file.endswith(".png"):
				#Get path to current file
				imageName = os.path.join(root, file)
				#Get parent folders name (needed for classification)
				parentFolder = os.path.basename(os.path.dirname(imageName))
				#Add folder name and path to list
				pathAndClassification = [parentFolder, imageName]
				#Create list to hold classification and data
				imageList.append(pathAndClassification)
				
	#Randomize the order of imageList		
	random.shuffle(imageList, random.random)

	#Create training data and write values from image
	i = 0
	print("Parsing files...")
	with open(savePath, "a") as trainingData:
		for name in imageList:
			#Create image object
			image = Image.open(name[1], "r")
			#Classification is first element in 2d array
			classification = int(name[0], 16)

			########################################################################################## IMPORTANT FOR FOLDER STRUCTURE
			#Convert folder name from hexadecimal to ascii (only use this if folder is in hexadecimal)
			#classification = chr(classification)
			########################################################################################## IMPORTANT FOR FOLDER STRUCTURE

			#Create backup image so original is not altered
			imageBackup = image.convert("L")
			imageBackup = imageBackup.resize((scaleSize, scaleSize), resample=Image.BICUBIC) #Image.LANCZOS
			imageBackup = PIL.ImageOps.invert(imageBackup)

			#Assign pixels from image
			pixels = list(imageBackup.getdata())
			#Set first value as correct classification
			trainingData.write(str(classification)+",")
			for x in range(1, len(pixels)+1):
				if x == len(pixels):
					trainingData.write(str(pixels[x-1]))
				else:
					trainingData.write(str(pixels[x-1]))
					trainingData.write(",")
			trainingData.write("\n")
			i += 1

def getopts(argv):
    opts = {}
    while argv:
        if argv[0][0] == '-':
            opts[argv[0]] = argv[1]
        argv = argv[1:]
    return opts

if __name__ == '__main__':
	#Get working directory
	rootDir = os.getcwd()

	myargs = getopts(argv)
	if '-p' in myargs and '-s' in myargs:
		rootDir = myargs['-p']
		saveDir = myargs['-s']
		scale = int(input("Scale size: "))
		createCsv(rootDir, saveDir, scale)

	else:
		print("Incorrect arguments")