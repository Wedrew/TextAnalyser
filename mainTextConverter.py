import numpy as np
from paper import *

#   The main file utilizing paper.py. Converts a given image into text.

text = ""
paper = Paper()
for line in range(paper.numLines):
    words = paper.partitionLine(line)
    for word in words:
        strWord = paper.partitionWord(line,word)
        text = text + strWord + "\n"
