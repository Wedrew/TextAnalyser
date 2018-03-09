import numpy
import os
import sys
import scipy.misc
import src.graphicshelper
from src.helper import *
from models.network import NeuralNetwork

def loadMenu(rootDir):

	#info = getComputerInfo()

	#Load the mnist test data CSV file into a list
	#CSV needs to be formated as such - First element in csv is the numbers value
	#then everything after corresponds to the images pixel value 0-255 separated by commas
	#a new line indicates a new image
	testDataFile = open(rootDir + "/data/testing/mnist_test.csv", 'r')
	testDataList = testDataFile.readlines()
	testDataFile.close()

	#Load the mnist training data CSV file into a list
	#This is where you should 
	trainingDataFile = open(rootDir + "/data/training/mnist_train.csv", 'r')
	trainingDataList = trainingDataFile.readlines()
	trainingDataFile.close()

	while True:
		#Ask whether to train or load network
		menuSelection = input("Train, load or quit: ")
		if menuSelection.lower() == "train":
			#User input validation
			while True:
				try:
					#Declare neural network parameters
					inputNodes = int(input("Input nodes (total pixels in image): "))
					assert inputNodes > 0
					hiddenNodes = int(input("Hidden nodes: "))
					assert hiddenNodes > 0
					outputNodes = int(input("Output nodes: "))
					assert outputNodes > 0
					learningRate = float(input("Learning rate: "))
					assert learningRate > 0
					epochs = int(input("Amount of epochs: "))
					assert epochs > 0
					break;
				except (ValueError, AssertionError, EOFError) as e:	
					print("Try again")

			printFiles(rootDir)

			trainingDataFile = input("Enter name of training data: ")
			if os.path.exists(rootDir + "/data/training/" + trainingDataFile + ".csv"):
				#Load the mnist training data CSV file into a list
				#This is where you should 
				trainingDataFile = open(rootDir + "/data/training/" + trainingDataFile + ".csv", 'r')
				trainingDataList = trainingDataFile.readlines()
				trainingDataFile.close()
			else:
				print("Does not exist")

			#Declare neural network
			nn = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
			#Train neural network
			nn.epoch(epochs, trainingDataList)
			print("Network trained")


			testDataFile = open(rootDir + "/data/testing/mnist_test.csv", 'r')
			testDataList = testDataFile.readlines()
			testDataFile.close()

			#After model is trained test data
			prompt = input("Model trained... \nPress any key to test")
			nn.testBatch(testDataList)
			while True:
				#Ask to save neural network
				save = input("Save network? (yes/no): ")

				if save.lower() == "yes":
					nn.save(rootDir)
					break
				elif save.lower() == "no":
					break
				else:
					print("Try again")
			break;
		elif menuSelection.lower() == "load":	
			#Declare neural network ()
			nn = NeuralNetwork()

			#Load data into model
			nn.load(rootDir)

			#Letter must be an array
			data = []

			#After model is trained test data
			nn.testLetter(data)
		elif menuSelection.lower() == "quit":
			break;
		else:
			print("Try again")