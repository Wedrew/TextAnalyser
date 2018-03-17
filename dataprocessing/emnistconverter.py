import numpy as np
import os
#Emnist conversion script that takes emnist data rotates it 90 degrees left, and flips the horizontal axis.
#Not sure why they provided the training data this way
rootDir = os.getcwd()
fileToConvert = rootDir + "/data/testing/emnist_letters_test"

with open(fileToConvert + ".csv", "r") as trainingDataFile:
	for record in trainingDataFile:
		data = record.split(",")
		correctLabel = int(data[0])
		del data[0]

		data = [int(x) for x in data]
		arrayData = np.asarray(data)
		arrayData = arrayData.reshape((28,28))
		arrayData = np.rot90(arrayData, 3)
		arrayData = np.fliplr(arrayData)
		arrayData = arrayData.flatten()
		data = arrayData.tolist()
		data.insert(0, correctLabel)

		with open(fileToConvert + "_converted.csv", "a") as convertedDataFile:
			y = 1
			for x in data:
				if y == len(data):
					convertedDataFile.write(str(x))
				else:
					convertedDataFile.write(str(x)+",")
					y += 1
			convertedDataFile.write('\n')