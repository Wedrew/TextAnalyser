import numpy as np
import random
import time
import scipy.ndimage
import matplotlib.pyplot
import os
#Emnist conversion script that takes emnist data rotates the picture 10 degrees to the left and right
#You will end up with 3x the amount of data
def augmentData(rooFile, saveFile)
	dataData = []
	with open(rootFile, "r") as trainingDataFile:
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
			dataData.append(data10DegreesRight)
			dataData.append(data10DegreesLeft)
			dataData.append(data)

	random.shuffle(dataData, random.random)

	with open(saveFile, "a") as augmentedDataFile:
		for x in range(len(fileData)):
			for y in range(len(fileData[x])):
				if y+1 == len(fileData[x]):
					augmentedDataFile.write(str(fileData[x][y]))
				else:
				augmentedDataFile.write(str(fileData[x][y])+",")

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
		augmentData(rootFile, saveFile)
	else:
		print("Incorrect arguments")