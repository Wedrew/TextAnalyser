import numpy
import os
import sys
import scipy.misc
from src.graphicshelper import GetComputerInfo
from src.helper import *
from models.network import NeuralNetwork

def loadMenu(rootDir):
	info = GetComputerInfo()
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

			while True:
				try:
					printFiles(rootDir + "/data/training/")
					trainingDataFile = input("Enter name of training file: ")
					assert trainingDataFile != ""
					if os.path.exists(rootDir + "/data/training/" + trainingDataFile):
						trainingDataFile = open(rootDir + "/data/training/" + trainingDataFile, 'r')
						trainingDataList = trainingDataFile.readlines()
						trainingDataFile.close()
						#Declare neural network
						nn = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
						#Train neural network
						nn.epoch(epochs, trainingDataList)
						print("Network trained")
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
						break
					elif (trainingDataFile.lower() == "quit"):
						break
					else:
						print("Does not exist")
				except (EOFError, AssertionError) as e:
					print("Try again")
		elif menuSelection.lower() == "load":	
			while True:
				nn = NeuralNetwork()
				nn.load(rootDir)
				try:
					printFiles(rootDir + "/data/testing/")
					testingDataFile = input("Enter name of testing file: ")
					assert testingDataFile != ""
					if os.path.exists(rootDir + "/data/testing/" + testingDataFile):
						testingDataFile = open(rootDir + "/data/testing/" + testingDataFile, 'r')
						testingDataList = testingDataFile.readlines()
						testingDataFile.close()
						nn.testBatch(testingDataList)
						break
					elif (testingDataFile.lower() == "quit"):
						break
					else:
						print("Try again")
				except (EOFError, AssertionError) as e:
					print("Try again")
		elif menuSelection.lower() == "quit":
			break
		else:
			print("Try again")
