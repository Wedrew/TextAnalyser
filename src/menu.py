import numpy
import os
import sys
import time
import scipy.misc
from src.graphicshelper import opencl
from src.helper import *
from src.paper import *
from src.graphicshelper import opencl
from models.network import NeuralNetwork

def loadMenu(rootDir):
	#info = GetComputerInfo()
	while True:
		#Ask whether to train or load network
		menuSelection = input("Train, load, convert, draw or ctrl-c to quit: ")
		if menuSelection.lower() == "train":
			#User input validation
			inputNodes = getInput("Input nodes(total pixels in image): ", dtype="i")
			hiddenNodesLOne = getInput("Hidden nodes in layer one: ", dtype="i")
			hiddenNodesLTwo = getInput("Hidden nodes in layer two: ", dtype="i")
			outputNodes = getInput("Output nodes: ", dtype="i")
			learningRate = getInput("Learning rate: ", dtype="f")
			epochs = getInput("Amount of epochs: ", dtype="i")

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
						nn = NeuralNetwork(inputNodes, hiddenNodesLOne, hiddenNodesLTwo, outputNodes, learningRate)
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
			nn = NeuralNetwork()
			if (nn.load(rootDir) != False):
				while True:
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
		elif menuSelection.lower() == "convert":
			network = NeuralNetwork()
			network.load(rootDir)

			text = ""
			paper = Paper()
			for line in range(paper.numLines):
			    print("Starting line %s!" %(line+1))
			    words = paper.partitionLine(line)
			    for word in words:
			        strWord = paper.partitionWord(line,word, network)
			        text = text + strWord + "\n"
		elif menuSelection.lower() == "draw":
			nn = NeuralNetwork()
			nn.load(rootDir)

			with open(rootDir + "/data/images/character.txt", "r") as imageFile:
				for record in imageFile:
					data = record.split(",")
					data = [int(x) for x in data]
					answer = nn.testLetter(data)

		elif menuSelection.lower() == "quit":
			break
		else:
			print("Try again")