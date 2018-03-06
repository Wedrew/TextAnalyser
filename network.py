import numpy
import math
import os
import time
import matplotlib.pyplot as plt
import scipy.special

#Maps letters to numbers for input to neural network
numLetterDict = {
    'a':10,
    'b':11,
    'c':12,
    'd':13,
    'e':14,
    'f':15,
    'g':16,
    'h':17,
    'i':18,
    'j':19,
    'k':20,
    'l':21,
    'm':22,
    'n':23,
    'o':24,
    'p':25,
    'q':26,
    'r':27,
    's':28,
    't':29,
    'u':30,
    'v':31,
    'w':32,
    'x':33,
    'y':34,
    'z':35,
    'A':36,
    'B':37,
    'C':38,
    'D':39,
    'E':40,
    'F':41,
    'G':42,
    'H':43,
    'I':44,
    'J':45,
    'K':46,
    'L':47,
    'M':48,
    'N':49,
    'O':50,
    'P':51,
    'Q':52,
    'R':53,
    'S':54,
    'T':55,
    'U':56,
    'V':57,
    'W':58,
    'X':59,
    'Y':60,
    'Z':61,
}

#Neural network definition
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate=0.0):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lrate = learningrate

        #Initailize matrices for weights between the hidden layer and input layer as well as 
        #the output layer and the hidden later
        self.weightInputHidden = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.weightHiddenOutput = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

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

        self.weightHiddenOutput += self.lrate * numpy.dot((outputErrors * finalOutputs * (1.0-finalOutputs)), numpy.transpose(hiddenOutputs))
        self.weightInputHidden += self.lrate * numpy.dot((hiddenErrors * hiddenOutputs * (1.0-hiddenOutputs)), numpy.transpose(inputs))

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

    def epoch(self, epochs, trainingDataList, outputNodes):
        #Train the neural network
        #Epochs is the number of times the training data set is used for training
        print("Training...")
        for e in range(epochs):
            #Go through all records in the training data set
            for record in trainingDataList:
                #Split the record by the ',' commas
                allValues = record.split(',')
                #Scale and shift the inputs
                inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
                #Create the target output values (all 0.01, except the desired label which is 0.99)
                targets = numpy.zeros(outputNodes) + 0.01
                #All_values[0] is the target label for this record
                targets[int(allValues[0])] = 0.99
                self.train(inputs, targets)
                pass
            pass
        print("Done training...")

    def test(self, testDataList):
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

            #Test if nn was correct
            if (label == correctLabel):
                #Network's answer matches correct answer
                correct = correct + 1
            else:
                #Network's answer does not match correct answer
                #Show incorrectly guess image
                pass
                # allValues = testDataList[showImageNum].split(',')
                # imageArray= numpy.asfarray(allValues[1:]).reshape((28,28))
                # matplotlib.pyplot.imshow(imageArray, cmap='Greys', interpolation='None')
                # matplotlib.pyplot.show()
            
            showImageNum += 1

        print("Performance: {}%".format((correct/sizeDataList)*100))

    def load(self, wih, who): #Check if folder containing all weights exists and load it
        #Assign loaded network
        self.weightInputHidden = wih
        self.weightHiddenOutput = who

    def save(self): #Save weights into folder using networkname
        numpy.save("layer1.npy", self.weightInputHidden)
        numpy.save("layer2.npy", self.weightHiddenOutput)
        print("Networks successfully saved")

    def softmax(self, x): #Compute softmax
        return numpy.max(numpy.exp(x) / float(sum(numpy.exp(x))))
