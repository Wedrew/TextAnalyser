import numpy
import math
import matplotlib.pyplot
import scipy.special
import time

#Neural network definition
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
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
        print("Error:", hidden_errors)

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

#Declare neural network parameters
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = float(input("Learning rate: "))

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#Train the neural network
#Epochs is the number of times the training data set is used for training
epochs = int(input("Amount of epochs: "))
time_start = time.time()
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
        n.train(inputs, targets)
        pass
    pass

print("Done training...")
time_elapsed = time.time()-time_start
print("Program took {:d} minutes and {:.2f} seconds".format(int(time_elapsed//60), time_elapsed%60))
prompt = input("Press any key test trained model")

#Load the mnist test data CSV file into a list
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#Scorecard for how well the network performs, initially empty
scorecard = []
correct = 0
num = 0
print("Testing data...")
#Go through all the records in the test data set
for record in test_data_list:
    #Split the record by the ',' commas
    all_values = record.split(',')
    #Correct answer is first value
    correct_label = int(all_values[0])
    #Scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #Query the network
    outputs = n.query(inputs)
    #The index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print("Network's answer:", label)
    print("Correct label:", correct_label)
    
    # Display correct letter
    #all_values = test_data_list[num].split(',')
    #image_array= numpy.asfarray(all_values[1:]).reshape((28,28))
    #matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='none')
    #matplotlib.pyplot.show()
    #num = num+1

    #Append correct or incorrect to list
    if (label == correct_label):
        #Network's answer matches correct answer, add 1 to scorecard
    	scorecard.append(1)
    	correct = correct + 1
    else:
        #Network's answer doesn't match correct answer, add 0 to scorecard
    	scorecard.append(0)
    	pass
    pass

print(correct/10000)
