import network
import numpy
import sys
from network import neuralNetwork
import scipy.misc

def loadMenu():
	#Load the mnist test data CSV file into a list
	#CSV needs to be formated as such - First element in csv is the numbers value
	#then everything after corresponds to the images pixel value 0-255 separated by commas
	#a new line indicates a new image
	testDataFile = open("mnist_test.csv", 'r')
	testDataList = testDataFile.readlines()
	testDataFile.close()

	#Load the mnist training data CSV file into a list
	#This is where you should 
	trainingDataFile = open("mnist_train.csv", 'r')
	trainingDataList = trainingDataFile.readlines()
	trainingDataFile.close()

	#Declare neural network parameters
	inputNodes = 784
	hiddenNodes = 300
	outputNodes = 10


	#Ask whether to train or load network
	train = input("Train or load network?: ")
	if train.lower() == "train":
		learningRate = float(input("Learning rate: "))

		#Declare neural network
		nn = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

		#Get epochs and train network
		epochs = int(input("Amount of epochs: "))
		nn.epoch(epochs, trainingDataList)
		prompt = input("Model trained... \nPress any key to test")

		#After model is trained test data
		nn.test(testDataList)

		#Ask to save neural network
		save = input("Save network?: ")
		if save.lower() == "yes":
			nn.save()

	elif train.lower() == "load":
		#Load connection weights
		wih = numpy.load("layer1.npy")
		who = numpy.load("layer2.npy")

		#Declare neural network
		nn = neuralNetwork(inputNodes, hiddenNodes, outputNodes)

		#Load data into model
		nn.load(wih, who)

		#After model is trained test data
		nn.test(testDataList)