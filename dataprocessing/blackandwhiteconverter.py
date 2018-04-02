import numpy as np
import os
import time
#Emnist conversion script that turns images from grayscale to only black and white
rootDir = os.getcwd()
fileToConvert = rootDir + "/data/training/mnist_train"

images = []
start_time = time.time()
with open(fileToConvert + ".csv", "r") as trainingDataFile:
	for record in trainingDataFile:
		data = record.split(",")
		correctLabel = int(data[0])
		del data[0]
		data = [int(x) for x in data]

		

		images.append(data)

		# with open(fileToConvert + "_baw" + ".csv", "a") as convertedDataFile:
		# 	y = 1
		# 	for x in data:
		# 		if y == len(data):
		# 			convertedDataFile.write(str(x))
		# 		else:
		# 			convertedDataFile.write(str(x)+",")
		# 			y += 1
		# 	convertedDataFile.write('\n')

# your code
elapsed_time = time.time() - start_time
print("Time:{}".format(elapsed_time))