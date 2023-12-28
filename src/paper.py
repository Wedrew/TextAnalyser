from .paperHelper import *
from PIL import Image
import PIL.ImageOps
import numpy as np
import time
import math

#   This file contains most code sufficient to open a specified image file
#   and write the contents into a .txt file. Relies on a neural network in
#   a separate file.

#   A class representing a scanned notebook paper containing pixel data,
#   margin width, starting line height, line spacing, and number of lines
class Paper(object):
    def __init__(self):
        self.pixels, self.image = getImage('data/images/testPaper.jpg')
        self.margin = getMargin(self)
        self.lineH, self.topLine = getLineData(self)
        self.numLines = (int)((self.pixels.shape[0] - self.topLine) / self.lineH)-1

    #   Given the paper, returns a list of length 2 arrays [start, end] denoting
    #   the "y" coordinates of the beginning and end of detected words in a line
    #   on the paper. The line number is passed as a parameter.
    def partitionLine(self, line):
        yCoords = []
        tol = 3         # num pixels right of margin

        #   Contains (h,w) coords of current position in line: begins just above
        #   bottom line
        start = [self.topLine + (line+1) * self.lineH - 1, self.margin + tol]
        print("Bottom of line computed to be at %s" %start[0])
        
        end = self.pixels.shape[1]     # end of line
        #   Until the end of the line, check for next word then next whitespace
        #   Add the y coords of the starts of new words/ whitespaces
        while (start[1] < end):
            start[1] = searchLine(self,line,start,1)
            if start[1] != end:
                yCoords.append(start[1]-2) # adds buffer to beginning of words
            else:
                break
            start[1] = searchLine(self,line,start,0)
            if start[1] != end:
                yCoords.append(start[1])

        #   For nonempty lines, "yCoords" should have odd length iff the line ends
        #   with a word (no extra space) and even length iff there's extra whitespace.
        #   Adds end bound to the last word in the former case
        if len(yCoords) == 0:
            print("Found line %s to be empty." %(line+1))
            return []
        elif len(yCoords)%2 == 1:
            yCoords.append(end)

        #   Formats yCoords to have desired [start,end] structure for words in line
        words = []
        for i in range(0,len(yCoords)-1,2):     # jump of 2 skips over whitespaces
            words.append([yCoords[i],yCoords[i+1]])
        print("Successfully read line %s; found %s words." %(line+1,len(words)))
        print("The words array is ", words)

        return words

    #   Breaks words into pieces of length < maxLen + 2*minLen to ensure the
    #   asymptotically expensive (but superior) wordBitToLetters() method deals with
    #   a reasonable number of potential partitions. Recursively splits the interval
    #   by checking what partition leads to the best letters on each side.
    def partitionWord(self, points, A, B, start, end, netOutputs):
        minLen = 4
        maxLen = 28
        maxConf = 0
        if B-A >= maxLen + 2*minLen:
            #   Determine bounds for partition
            a = int((A + B - maxLen) / 2)
            b = int((A + B + maxLen) / 2)
            cMax = a

            #   For each possible partition, try all possible letters to the left
            #   and right of partition; the combination yielding the highest total
            #   confidence determines what the best value for c is (cMax).
            for c in range(a, b):
                #   Make sure letter choices don't go out of bounds [A, B]
                for leftLen in range(minLen,min(maxLen, c - A)):
                    for rightLen in range(minLen, min(maxLen, B - c)):
                        conf = 0
                        conf += netOutputs[leftLen-minLen][c-leftLen-start][0]
                        conf += netOutputs[rightLen-minLen][c-start][0]
                        if conf > maxConf:
                            maxConf = conf
                            cMax = c
                    
            points.add(cMax)
            points.update(self.partitionWord(set(points), A, cMax, start, end, netOutputs))
            points.update(self.partitionWord(set(points), cMax, B, start, end, netOutputs))
            return points
        else:
            return [A,B]
        
    #   Given a word, the paper, current line, and a NN to parse the word,
    #   returns the word as a single string
    def wordBitToLetters(self, line, word, netOutputs):
        #   Define borders of the entire word
        bot = self.lineH * (line+1) + self.topLine
        top = self.lineH * line + self.topLine
        start = word[0]
        end = word[1]
        w = end - start

        #   Define range of legitimate lengths for letters
        minLen = 4
        maxLen = 28

        #   Define range of how many letters could be in word
        maxLet = int(w/minLen)
        minLet = max(1,math.ceil(w/maxLen))
        #print("There will be between %s and %s letters." %(minLet,maxLet))

        #   For each possible number of letters, test all permutations, or choice
        #   of partitioning of those letters
        maxConf = 0
        bestVec = []
        for numLetts in range(minLet,maxLet+1):
            #print("At %s letters:" %numLetts)
            #   Establish the "zero vector": a vector representing the first
            #   possible permutation under the indexing system
            zeroVec = []
            start = word[0]
            for i in range(numLetts):
                #   Define lowest current letter sizes that guarantees existence
                #   of at least one set of letters of legitimate length that
                #   complete the word
                minLi = max(minLen,w - (numLetts - i - 1)*maxLen)
                zeroVec.append(minLi)
                w -= minLi
            w = end - start     # want to reuse w with its original value later

            posVec = zeroVec    # the vector corresponding to current permutation
            numPerms = 0
            #print("w is", w)
            #print("posVec = zeroVec =", posVec)
            while posVec[0] != 0:
                #   Generate word at current "position vector"
                currWord = ""
                conf = 0
                pos = start
                for i in range(numLetts):
                    #   See getImageCache if indexing is confusing
                    confAndLetter = netOutputs[posVec[i]-minLen][pos-start]
                    conf += confAndLetter[0]
                    letter = str(confAndLetter[1])
                    currWord += letter
                    pos += posVec[i]
                
                #   Save word and confidence that corresponded to it if it's the best
                if conf / numLetts > maxConf:
                    maxConf = conf / numLetts
                    finalWord = currWord
                    bestVec = list(posVec)    # to help debug which vector is most effective
                
                #   Iterate to next permutation
                posVec = addTwo(posVec,numLetts,minLen,maxLen,w)
                numPerms += 1
                
            #print("Number of permutations so far:", numPerms)
            #print("Best word so far had vector", bestVec)
            #print("The word was", finalWord)

        print("Best bit overall had vector", bestVec)
        #print("The word was", finalWord)
        #print()
        return finalWord
