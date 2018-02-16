import network
import numpy
from network import neuralNetwork

#Load the mnist test data CSV file into a list
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#Load the mnist training data CSV file into a list
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#Declare neural network parameters
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = float(input("Learning rate: "))

#Declare neural network
nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#Get epochs and train network
epochs = int(input("Amount of epochs: "))
nn.epoch(epochs, training_data_list, output_nodes)
prompt = input("Model trained... \nPress any key to test")

#After model is trained test data
nn.testWithoutImage(test_data_list)

#Ask to save neural network
save = input("Save neural network?: ")
if save.lower() == "yes":
	nn.save()