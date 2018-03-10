import numpy
import math
import os
import time
import matplotlib.pyplot
import scipy.special
import PIL
import string
from datetime import datetime
from sys import exit
import pyopencl as cl
import pyopencl.array as pycl_array
from src.helper import *

#Mapping for emnist training data, does not include some letters which may have similar
#lower and upper case for better performance
#Converting from decimal to ascii is done in epoch function
emnistMapping = {
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9',
    10:'A',
    11:'B',
    12:'C',
    13:'D',
    14:'E',
    15:'F',
    16:'G',
    17:'H',
    18:'I',
    19:'J',
    20:'K',
    21:'L',
    22:'M',
    23:'N',
    24:'O',
    25:'P',
    26:'Q',
    27:'R',
    28:'S',
    29:'T',
    30:'U',
    31:'V',
    32:'W',
    33:'X',
    34:'Y',
    35:'Z',
    36:'a',
    37:'b',
    38:'d',
    39:'e',
    40:'f',
    41:'g',
    42:'h',
    43:'n',
    44:'q',
    45:'r',
    46:'t',
}

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
        #Heavy lifting is done here
        #Will rewrite this using opencl
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
            correctLabel = emnistMapping[int(allValues[0])]
            #Scale and shift the inputs
            inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
            #Query the network
            outputs = self.query(inputs)
            #The index of the highest value corresponds to the label
            label = emnistMapping[numpy.argmax(outputs)]
            #Produces the softmax "probability" of the network
            certainty = self.softmax(self.softmaxOutputs)
            print("-----------------------------")
            print("Network's answer:", label)
            print("Correct label:", correctLabel)
            print("Certainty: {}%".format(certainty*100))
            print("-----------------------------")
            
            # #Uncomment to see text
            allValues = testDataList[showImageNum].split(',')
            imageArray= numpy.asfarray(allValues[1:]).reshape((28,28))
            matplotlib.pyplot.imshow(imageArray, cmap='Greys', interpolation='None')
            matplotlib.pyplot.show()

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
        printFiles(rootDir + "/savednetworks/")
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
                print("Network successfully loaded")
                break
            elif networkName.lower() == "quit":
                break
            else:
                print("Network name did not exist")
                networkName = input("Saved network name: ")

    def save(self, rootDir): #Save weights into folder using networkname
        #Create folders and files
        while True:
            networkName = input("What would you like to save network as?: ")
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
            elif networkName.lower() == "quit":
                break
            else:
                print("Network name already exists")
        print("Networks successfully saved")

    def softmax(self, x): #Compute softmax
        return numpy.max(numpy.exp(x) / float(sum(numpy.exp(x))))

    def setNetwork(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate