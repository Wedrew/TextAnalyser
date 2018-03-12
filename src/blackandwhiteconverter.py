import numpy as np
import os
import time
#Emnist conversion script
rootDir = os.getcwd()
fileToConvert = rootDir + "/data/testing/emnist_test"

with open(fileToConvert + ".csv", "r") as trainingDataFile:
	for record in trainingDataFile:
		data = record.split(",")
		correctLabel = int(data[0])
		del data[0]
		data = [int(x) for x in data]

		counter = 0
		for x in data:
			if x > 0:
				data[counter] = 255
				counter += 1
			else:
				counter += 1

		data.insert(0, correctLabel)

		with open(fileToConvert + "_baw" + ".csv", "a") as convertedDataFile:
			y = 1
			for x in data:
				if y == len(data):
					convertedDataFile.write(str(x))
				else:
					convertedDataFile.write(str(x)+",")
					y += 1
			convertedDataFile.write('\n')