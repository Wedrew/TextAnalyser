import numpy as np
from src.paper import *
from models.network import *
import os

#   The main file utilizing paper.py. Converts a given image into text.

network = NeuralNetwork()
network.load(".")

text = ""
paper = Paper()
for line in range(paper.numLines):
    print("Starting line %s!" %(line+1))
    words = paper.partitionLine(line)
    for word in words:
        strWord = paper.partitionWord(line,word, network)
        text = text + strWord + "\n"
