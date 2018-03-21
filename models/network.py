import numpy
import math
import os
import time
import matplotlib.pyplot
import scipy.special
import PIL
import string
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.tools as cl_tools
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sys import exit
from src.graphicshelper import opencl
from skimage.feature import hog
from data.mappings.mappings import *
from src.helper import *

#Neural network definition
class NeuralNetwork:
    def __init__(self, inputNodes=1, hiddenNodesLOne=1, hiddenNodesLTwo=1, outputNodes=1, learningRate=0.0):
        self.inputNodes = inputNodes
        self.hiddenNodesLOne = hiddenNodesLOne
        self.hiddenNodesLTwo = hiddenNodesLTwo
        self.outputNodes = outputNodes
        self.learningRate = learningRate
        #Initailize matrices for weights between the hidden layers and input layer as well as 
        #the output layer and the hidden later
        self.weightInputHidden = numpy.random.normal(0.0, pow(self.hiddenNodesLOne, -0.5), (self.hiddenNodesLOne, self.inputNodes))
        self.weightHiddenLOneHiddenLTwo = numpy.random.normal(0.0, pow(self.hiddenNodesLTwo, -0.5), (self.hiddenNodesLTwo, self.hiddenNodesLOne))
        self.weightHiddenLTwoOutput = numpy.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodesLTwo))
        #Activation function
        self.activationFunction = lambda x: scipy.special.expit(x)

    def train(self, inputsList, targetsList):
        #Heavy lifting is done here
        #Will rewrite this using opencl/cudnn
        inputs = numpy.array(inputsList, ndmin=2, dtype=numpy.float32).T 
        targets = numpy.array(targetsList, ndmin=2, dtype=numpy.float32).T 
        #First hidden layer
        hiddenInputs = numpy.dot(self.weightInputHidden, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)
        #Second hidden layer
        hiddenLOne = numpy.dot(self.weightHiddenLOneHiddenLTwo, hiddenOutputs)
        hiddenLTwo = self.activationFunction(hiddenLOne)
        #Final output layer
        finalInputs = numpy.dot(self.weightHiddenLTwoOutput, hiddenLTwo)
        finalOutputs = self.activationFunction(finalInputs)
        #Calculate error (target-actual)
        outputErrors = (targets-finalOutputs)
        #Errors for two hidden layers
        hiddenLTwoErrors = numpy.dot(self.weightHiddenLTwoOutput.T, outputErrors)
        hiddenLOneErrors = numpy.dot(self.weightHiddenLOneHiddenLTwo.T, hiddenLTwoErrors)
        #Update weights
        self.weightHiddenLTwoOutput += self.learningRate * numpy.dot((outputErrors * finalOutputs * (1.0-finalOutputs)), (hiddenLTwo).T)
        self.weightHiddenLOneHiddenLTwo += self.learningRate * numpy.dot((hiddenLTwoErrors * hiddenLTwo * (1.0-hiddenLTwo)), (hiddenOutputs).T)
        self.weightInputHidden += self.learningRate * numpy.dot((hiddenLOneErrors * hiddenOutputs * (1.0-hiddenOutputs)), (inputs).T)

    def query(self, inputsList):
        #Convert inputs list to 2d array
        inputs = numpy.array(inputsList, ndmin=2).T
        #Calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.weightInputHidden, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)
        #Calculate signals into second hidden layer
        hiddenLOne = numpy.dot(self.weightHiddenLOneHiddenLTwo, hiddenOutputs)
        hiddenLTwo = self.activationFunction(hiddenLOne)
        #Calculate signals into final output later
        finalInputs = numpy.dot(self.weightHiddenLTwoOutput, hiddenLTwo)
        #Used to determine networks confidence
        self.softmaxOutputs = finalInputs
        #Calculate the signals emerging from the final layer
        finalOutputs = self.activationFunction(finalInputs)
        return finalOutputs

    def epoch(self, epochs, trainingDataList):
        #Train the neural network
        #Epochs is the number of times the training data set is used for training
        startTime = datetime.now()
        print("Training...")
        for e in range(epochs):
            #Go through all records in the training data set
            for record in trainingDataList:
                #Split the record by the ',' commas
                allValues = record.split(',')
                targetValue = (allValues[0])
                #Scale and shift the inputs
                inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
                #Create the target output values (all 0.01, except the desired label which is 0.99)
                targets = numpy.zeros(self.outputNodes) + 0.01
                #All_values[0] is the target label for this record
                targets[int(float(targetValue))] = 0.99
                #Train network
                self.train(inputs, targets)
                pass
            pass
        print("Done training...")
        print("Took {}".format(datetime.now() - startTime))

    def testBatch(self, testDataList):
        print("Testing data...")
        correct = 0
        sizeDataList = 0
        #Gets total digits to be entered
        for record in testDataList:
            sizeDataList += 1
        #Go through all the records in the test data set
        for record in testDataList:
            #Split the record by the ',' commas
            allValues = record.split(',')
            #Correct answer is first value
            correctLabel = emnistLetterMapping[int(float(allValues[0]))]
            #Scale and shift the inputs
            inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
            #Query the network
            outputs = self.query(inputs)
            #The index of the highest value corresponds to the label
            label = emnistLetterMapping[numpy.argmax(outputs)]
            #Produces the softmax "probability" of the network
            certainty = self.softmax(self.softmaxOutputs)
            print("******************************")
            print("Network's answer:", label)
            print("Correct label:", correctLabel)
            print("Certainty: {}%".format(certainty*100))
            print("******************************")

            # imageArray= numpy.asfarray(allValues[1:]).reshape((28,28))
            # matplotlib.pyplot.imshow(imageArray, cmap='Greys', interpolation='None')
            # matplotlib.pyplot.show()

            #Test if nn was correct
            if (label == correctLabel):
                #Network's answer matches correct answer
                correct += 1
            else:
                #Network's answer does not match correct answer
                pass
        print("Performance: {}%".format((correct/sizeDataList)*100))

    def testLetter(self, image):
        #Scale and shift inputs
        inputs = (numpy.asfarray(image[:]) / 255.0 * 0.99) + 0.01
        #Query network
        outputs = self.query(inputs)
        #Index of the highest value corresponds to the label
        label = emnistLetterMapping[numpy.argmax(outputs)]
        #Calculate certainty
        certainty = self.softmax(self.softmaxOutputs)
        #Print out predicted number
        print("This is a {}".format(label))
        print("Certainty: {}%".format(certainty*100))
        # imageArray= numpy.asfarray(image[:]).reshape((28,28))
        # matplotlib.pyplot.imshow(imageArray, cmap='Greys', interpolation='None')
        # matplotlib.pyplot.show()
        return (certainty, label)

    def load(self, rootDir):
        #Assign loaded network
        while True:
            try:
                printFolders(rootDir + "/savednetworks/")
                networkName = input("Enter name of saved network: ")
                assert networkName != ""
                #Check to see if folder for networkname exists
                if os.path.exists(rootDir + "/savednetworks/" + networkName):
                    #Load files from folder
                    layer1 = numpy.load(rootDir + "/savednetworks/" + networkName + "/layer1.npy")
                    layer2 = numpy.load(rootDir + "/savednetworks/" + networkName + "/layer2.npy")
                    layer3 = numpy.load(rootDir + "/savednetworks/" + networkName + "/layer3.npy")
                    self.weightInputHidden = layer1
                    self.weightHiddenLOneHiddenLTwo = layer2
                    self.weightHiddenLTwoOutput = layer3

                    with open(rootDir + "/savednetworks/" + networkName + "/info.txt", "r") as textFile:
                        for line in textFile:
                            fields = line.split(",")
                            self.setNetwork(fields[0], fields[1], fields[2], fields[3], fields[4])
                    print("Network successfully loaded")
                    return True
                elif networkName.lower() == "quit":
                    return False
                else:
                    print("Network name did not exist")
            except (EOFError, AssertionError):
                    print("Try again")

    def save(self, rootDir): #Save weights into folder using networkname
        #Create folders and files
        while True:
            try:
                networkName = input("What would you like to save network as?: ")
                assert networkName != ""
                #Check to see if folder for networkname already exists
                if not os.path.exists(rootDir + "/savednetworks/" + networkName):
                    #Create folder to hold networks
                    os.makedirs(rootDir + "/savednetworks/" + networkName)
                    #Populate folder with files
                    numpy.save(rootDir + "/savednetworks/" + networkName + "/layer1.npy", self.weightInputHidden)
                    numpy.save(rootDir + "/savednetworks/" + networkName + "/layer2.npy", self.weightHiddenLOneHiddenLTwo)
                    numpy.save(rootDir + "/savednetworks/" + networkName + "/layer3.npy", self.weightHiddenLTwoOutput)
                    #Create file to hold info about networks
                    with open(rootDir + "/savednetworks/" + networkName + "/info.txt", "w") as textFile:
                        textFile.write("{},{},{},{},{}".format(self.inputNodes, self.hiddenNodesLOne, self.hiddenNodesLTwo, self.outputNodes, self.learningRate))
                        textFile.close()
                        break
                elif networkName.lower() == "quit":
                    return
                else:
                    print("Network name already exists")
            except (EOFError, AssertionError):
                print("Try again")
        print("Networks successfully saved")

    def softmax(self, x): #Compute softmax
        return numpy.max(numpy.exp(x) / float(sum(numpy.exp(x))))

    def setNetwork(self, inputNodes, hiddenNodesLOne, hiddenNodesLTwo, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodesLOne = hiddenNodesLOne
        self.hiddenNodesLTwo = hiddenNodesLTwo
        self.outputNodes = outputNodes
        self.learningRate = learningRate