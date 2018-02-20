import numpy
import math
import os
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
        self.weight_input_hidden = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.weight_hidden_output = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        #Activation function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        #Error (target-actual)
        output_errors = targets - final_outputs

        hidden_errors = numpy.dot(self.weight_hidden_output.T, output_errors)

        self.weight_hidden_output += self.lrate * numpy.dot((output_errors * final_outputs * (1.0-final_outputs)), numpy.transpose(hidden_outputs))
        self.weight_input_hidden += self.lrate * numpy.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        #Convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        #Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #Calculate signals into final output later
        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)

        #Calculate the signals emerging from the final layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def epoch(self, epochs, training_data_list, output_nodes):
        #Train the neural network
        #Epochs is the number of times the training data set is used for training
        print("Training...")
        for e in range(epochs):
            #Go through all records in the training data set
            for record in training_data_list:
                #Split the record by the ',' commas
                all_values = record.split(',')
                #Scale and shift the inputs
                inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                #Create the target output values (all 0.01, except the desired label which is 0.99)
                targets = numpy.zeros(output_nodes) + 0.01
                #All_values[0] is the target label for this record
                targets[int(all_values[0])] = 0.99
                self.train(inputs, targets)
                pass
            pass
        print("Done training...")

    def test(self, test_data_list):
        print("Testing data...")
        correct = 0
        #Go through all the records in the test data set
        for record in test_data_list:
            #Split the record by the ',' commas
            all_values = record.split(',')
            #Correct answer is first value
            correct_label = int(all_values[0])
            #Scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            #Query the network
            outputs = self.query(inputs)
            #The index of the highest value corresponds to the label
            label = numpy.argmax(outputs)
            print("Network's answer:", label)
            print("Correct label:", correct_label)

            #Test if nn was correct
            if (label == correct_label):
                #Network's answer matches correct answer
                correct = correct + 1
            else:
                #Network's answer doesn't match correct answer
                pass
            pass
        print(correct/10000)

    def load(self, wih, who): #Check if folder containing all weights exists and load it
        #Assign loaded network
        self.weight_input_hidden = wih
        self.weight_hidden_output = who

    def save(self, path): #Save weights into folder using networkname
        numpy.save(str(path) + "layer1.npy", self.weight_input_hidden)
        numpy.save(str(path) + "layer2.npy", self.weight_hidden_output)
        print("Networks successfully saved")