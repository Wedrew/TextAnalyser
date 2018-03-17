import numpy as np
import random
import time
import scipy.ndimage
import matplotlib.pyplot
import os
#Emnist conversion script that takes emnist data rotates the picture 10 degrees to the left and right
#You will end up with 3x the amount of data
rootDir = os.getcwd()
fileToConvert = rootDir + "/data/testing/emnist_letters_test"

with open(fileToConvert + ".csv", "r") as trainingDataFile:
	totalFiles = []
	for record in trainingDataFile:
		data = record.split(",")
		correctLabel = int(data[0])
		del data[0]

		data = [int(x) for x in data]
		data = np.array(data, dtype=np.int)
		#Rotate images
		data10DegreesRight = scipy.ndimage.interpolation.rotate(data.reshape(28,28), 10, cval=0.01, reshape=False)
		data10DegreesLeft = scipy.ndimage.interpolation.rotate(data.reshape(28,28), -10, cval=0.01, reshape=False)
		#Flatten back to (784,)
		data10DegreesRight = data10DegreesRight.flatten()
		data10DegreesLeft = data10DegreesLeft.flatten()
		#Convert back to lists
		data10DegreesRight = data10DegreesRight.tolist()
		data10DegreesLeft = data10DegreesLeft.tolist()	
		data = data.tolist()

		#Reenter correct label
		data10DegreesRight.insert(0, correctLabel)
		data10DegreesLeft.insert(0, correctLabel)
		data.insert(0, correctLabel)

		#Make all values positive
		data10DegreesRight =[abs(x) for x in data10DegreesRight]
		data10DegreesLeft = [abs(x) for x in data10DegreesLeft]
		data = [abs(x) for x in data]

		#Append files to total files
		totalFiles.append(data10DegreesRight)
		totalFiles.append(data10DegreesLeft)
		totalFiles.append(data)

	random.shuffle(totalFiles, random.random)

	with open(fileToConvert + "_augmented.csv", "a") as augmentedDataFile:
		for x in totalFiles:
			pixels = x[:]
			p = 1
			for y in pixels:
				if p == len(pixels):
					augmentedDataFile.write(str(y))
				else:
					augmentedDataFile.write(str(y)+",")
				p+=1
			augmentedDataFile.write("\n")