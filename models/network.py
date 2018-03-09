import numpy
import math
import os
import matplotlib.pyplot as plt
import scipy.special
import PIL
import string
from datetime import datetime
from sys import exit
import pyopencl as cl
import pyopencl.array as pycl_array

#Maps letters to numbers for input to neural network
letterMappings = dict(zip(string.ascii_letters, range(1, 53)))

#Neural network definition
class NeuralNetwork:
    def __init__(self, inputNodes=1, hiddenNodes=1, outputNodes=1, learningRate=0.0):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate

        #Initailize matrices for weights between the hidden layer and input layer as well as 
        #the output layer and the hidden later
        self.weightInputHidden = numpy.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.weightHiddenOutput = numpy.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))

        #Activation function
        self.activationFunction = lambda x: scipy.special.expit(x)

    def train(self, inputsList, targetsList):
        inputs = numpy.array(inputsList, ndmin=2).T
        targets = numpy.array(targetsList, ndmin=2).T

        hiddenInputs = numpy.dot(self.weightInputHidden, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        finalInputs = numpy.dot(self.weightHiddenOutput, hiddenOutputs)
        finalOutputs = self.activationFunction(finalInputs)

        #Error (target-actual)
        outputErrors = targets - finalOutputs

        hiddenErrors = numpy.dot(self.weightHiddenOutput.T, outputErrors)

        self.weightHiddenOutput += self.learningRate * numpy.dot((outputErrors * finalOutputs * (1.0-finalOutputs)), numpy.transpose(hiddenOutputs))
        self.weightInputHidden += self.learningRate * numpy.dot((hiddenErrors * hiddenOutputs * (1.0-hiddenOutputs)), numpy.transpose(inputs))

    def query(self, inputsList):
        #Convert inputs list to 2d array
        inputs = numpy.array(inputsList, ndmin=2).T

        #Calculate signals into hidden layer
        hiddenInputs = numpy.dot(self.weightInputHidden, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)

        #Calculate signals into final output later
        finalInputs = numpy.dot(self.weightHiddenOutput, hiddenOutputs)

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

                #Scale and shift the inputs
                inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01

                #Create the target output values (all 0.01, except the desired label which is 0.99)
                targets = numpy.zeros(self.outputNodes) + 0.01

                #All_values[0] is the target label for this record
                targets[int(allValues[0])] = 0.99

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
        showImageNum = 0
        #Gets total digits to be entered
        for record in testDataList:
            sizeDataList += 1

        #Go through all the records in the test data set
        for record in testDataList:
            #Split the record by the ',' commas
            allValues = record.split(',')

            #Correct answer is first value
            correctLabel = int(allValues[0])

            #Scale and shift the inputs
            inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01

            #Query the network
            outputs = self.query(inputs)

            #The index of the highest value corresponds to the label
            label = numpy.argmax(outputs)

            #Produces the softmax "probability" of the network
            certainty = self.softmax(self.softmaxOutputs)

            print("-----------------------------")
            print("Network's answer:", label)
            print("Correct label:", correctLabel)
            print("Certainty: {}%".format(certainty*100))
            print("-----------------------------")
            
            #Uncomment to see text
            #allValues = testDataList[showImageNum].split(',')
            #imageArray= numpy.asfarray(allValues[1:]).reshape((28,28))
            # matplotlib.pyplot.imshow(imageArray, cmap='Greys', interpolation='None')
            # matplotlib.pyplot.show()

            #Test if nn was correct
            if (label == correctLabel):
                #Network's answer matches correct answer
                correct += 1
            else:
                #Network's answer does not match correct answer
                pass
                
            showImageNum += 1

        print("Performance: {}%".format((correct/sizeDataList)*100))

    def testLetter(self, image):
        #Scale and shift inputs
        inputs = (numpy.asfarray(image[:]) / 255.0 * 0.99) + 0.01

        #Query network
        outputs = self.query(inputs)

        #Index of the highest value corresponds to the label
        label = numpy.argmax(outputs)

        #Calculate certainty
        certainty = self.softmax(self.softmaxOutputs)

        #Print out predicted number
        print("This is a {}".format(label))
        print("Certainty: {}".format(certainty))

    def load(self, rootDir):
        networkName = input("Saved network name: ")
        #Assign loaded network
        while True:
            #Check to see if folder for networkname exists
            if os.path.exists(rootDir + "/savednetworks/" + networkName):
                #Load files from folder
                wih = numpy.load(rootDir + "/savednetworks/" + networkName + "/layer1.npy")
                who = numpy.load(rootDir + "/savednetworks/" + networkName + "/layer2.npy")
                self.weightInputHidden = wih
                self.weightHiddenOutput = who

                with open(rootDir + "/savednetworks/" + networkName + "/info.txt", "r") as textFile:
                    for line in textFile:
                        fields = line.split(",")
                        self.setNetwork(fields[0], fields[1], fields[2], fields[3])
                break;
            else:
                print("Network name did not exist\nPlease try again")
                print("""Type "quit" to quit """)
                networkName = input("Saved network name: ")
                if networkName.lower() == "quit":
                    exit(0)

    def save(self, rootDir): #Save weights into folder using networkname
        networkName = input("What would you like to save network as?: ")
        #Create folders and files
        while True:
            #Check to see if folder for networkname already exists
            if not os.path.exists(rootDir + "/savednetworks/" + networkName):
                #Create folder to hold networks
                os.makedirs(rootDir + "/savednetworks/" + networkName)

                #Populate folder with files
                numpy.save(rootDir + "/savednetworks/" + networkName + "/layer1.npy", self.weightInputHidden)
                numpy.save(rootDir + "/savednetworks/" + networkName + "/layer2.npy", self.weightHiddenOutput)

                #Create file to hold info about networks
                with open(rootDir + "/savednetworks/" + networkName + "/info.txt", "w") as textFile:
                    textFile.write("{},{},{},{}".format(self.inputNodes, self.hiddenNodes, self.outputNodes, self.learningRate))
                    textFile.close()
                    break
            else:
                print("Network name already exists.\nPlease try again.")
                networkName = input("What would you like to save network as?: ")
        print("Networks successfully saved")

    def softmax(self, x): #Compute softmax
        return numpy.max(numpy.exp(x) / float(sum(numpy.exp(x))))

    def setNetwork(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate