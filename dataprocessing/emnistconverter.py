import numpy as np
from sys import argv
import time
import os
#Emnist conversion script that takes emnist data rotates it 90 degrees left, and flips the horizontal axis.
#Not sure why they provided the training data this way
def emnistConvert(rootFile, saveFile)
	fileData = []
	with open(rootFile, "r") as trainingDataFile:
		for record in trainingDataFile:
			data = record.split(",")
			#Zero indexes the correct label
			correctLabel = int(data[0])
			correctLabel -= 1
			data[0] = str(correctLabel)

			# # Use this to flip array
			# data = [int(x) for x in data]
			# arrayData = np.asarray(data)
			# arrayData = arrayData.reshape((28,28))
			# arrayData = np.rot90(arrayData, 3)
			# arrayData = np.fliplr(arrayData)
			# arrayData = arrayData.flatten()
			# data = arrayData.tolist()
			# data.insert(0, correctLabel)

			fileData.append(data)

	with open(saveFile, "w") as convertedDataFile:
		for x in range(len(fileData)):
			for y in range(len(fileData[x])):
				if y+1 == len(fileData[x]):
					convertedDataFile.write(str(fileData[x][y]))
				else:
					convertedDataFile.write(str(fileData[x][y])+",")

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
		emnistConvert(rootFile, saveFile)
	else:
		print("Incorrect arguments")