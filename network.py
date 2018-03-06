import numpy
import math
import os
import time
import matplotlib.pyplot
import scipy.special

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
