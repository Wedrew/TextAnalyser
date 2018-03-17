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

    #   Given a word, the paper, current line, and a NN to parse the word,
    #   returns the word as a single string
    def partitionWord(self, line, word, network):
        #   Define borders of the entire word
        bot = self.lineH * (line+1) + self.topLine
        top = self.lineH * line + self.topLine
        start = word[0]
        end = word[1]
        w = end - start

        #   Define range of legitimate lengths for letters
        minLen = 4
        maxLen = 28

        #   Get all letters and confidences associated with the boxes this method
        #   will check
        netOutputs = getImageCache(self,network,minLen,maxLen,start,end,bot,top)

        #   Define range of how many letters could be in word
        maxLet = int(w/minLen)
        minLet = max(1,math.ceil(w/maxLen))
        print("There will be between %s and %s letters." %(minLet,maxLet))

        #   For each possible number of letters, test all permutations, or choice
        #   of partitioning of those letters
        maxConf = 0
        bestVec = []
        for numLetts in range(minLet,maxLet+1):
            print("At %s letters:" %numLetts)
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
                
            print("Number of permutations so far:", numPerms)
            print("Best word so far had vector", bestVec)
            print("The word was", finalWord)

        print("Best word overall had vector", bestVec)
        print("The word was", finalWord)
        print()

        return finalWord

#   "Add 2 to" index vector, representing moving to the next permutation (choice of
#   word partitioning) and return that vector, or return [0] if at the last permutation
def addTwo(vec,numLetts,minLen,maxLen, w):
    maxLi = []
    wOrig = w
    #   Fill maxLi, the list of maximal legitimate letter lengths from right to left
    for i in range(numLetts):
        maxLi.append(min(w - (numLetts - i - 1)*minLen,maxLen))
        w -= vec[i]
    w = wOrig

    #   Add one to vector or return [0] if we can't (and we reached the last permutation)
    keepGoing = True
    i = numLetts-1
    while keepGoing:
        #   Increase last vector index if possible; if not try previous index
        #   and continue this way if necessary before adding 1
        if vec[i]+2 > maxLi[i]:
            i -= 1
            #   No more indices left to add 1 to; this means we were at the last
            #   permutation and are done; returns [0] as a flag to signal completion
            if i < 0:
                return [0]
        else:
            vec[i] += 2
            #   Have to reset the tail of the vector to be at the minimum
            #   given the head's new value
            if i < numLetts-1:
                #   Get value for wi
                for c in range(i+1):
                    w -= vec[c]
                #   Set the tail to the proper minimum
                for c in range(i+1,numLetts):
                    vec[c] = max(minLen,w - (numLetts - c - 1)*maxLen)
                    w -= vec[c]
                    
            keepGoing = False

    return vec

            
# -------------------------------------------------------------------------------------
#
#   Helper functions ( not methods )
#
# -------------------------------------------------------------------------------------

#   Sends the neural network all images it will need in partitionWord, and
#   generates a (maxLen - minLen + 1) length list of lists, where each element
#   is a tuple (letter, confidence) from network.testLetter. The elements
#   of each array are ordered by image starting position from left to right.
def getImageCache(paper, network, minLen, maxLen, start, end, bot, top):
    netOutputs = []
    numCached = 0
    #   Different the minimum heights, at each width value, that contain text
    #   with minimal whitespace leftover on top (to later shrink box onto letters)
    tops = []
    for pix in range(start,end):
        currH = top + 4
        #   Moves down until we hit a black pixel and records the ending height
        a = 255
        while a == 255 and currH <= bot-4:
            currH += 2
            a = paper.pixels[currH][pix]
        tops.append(currH-2)

    for w in range(minLen, maxLen+1):
        currOutputs = []
        #   Loop through all starting positions for a given image width:
        for startPos in range(end - start - w + 1):
            boxTop = min(tops[startPos:startPos+w]) # min means highest pixel
            letterI = toImage(start+startPos,bot-2,start+startPos+w,boxTop,paper)
            netOut = network.testLetter(letterI)
            currOutputs.append(netOut)
            numCached += len(currOutputs)
        netOutputs.append(currOutputs)
    print("Cached %s images and NN queries" %(numCached))
    return netOutputs


#   Given the pixels of the Paper, returns the number of pixels the margin on the paper
#   takes up. Used to reduce processing time of lines.
def getMargin(paper):
    h = 5   # distance from top corner in both dimensions
    
    #   Totals the brightness of a horizontal line segment of 3 pixels, starting close
    #   from the top left. If any are black (so activ < 255*3), returns y coord of
    #   middle pixel (which should be the margin).
    start = h
    activ = 1000
    while (activ >= 765):    # 765 = 255*3
        activ = 0
        for i in range(3):
            activ += paper.pixels[h][start+i]
        start += 3  # jump size for start of segment
    start -= 2

    print("Found the margin! ~%s pixels from the left." %start)
    return start

#   Given the Paper, and assuming the upper and lower bounds of the first line
#   are clear or marked just to the left of the margin, returns a tuple:
#   (lineH, topL) where lineH is the distance between lines vertically on the
#   Paper, and topL is the distance from the top of the paper to the top of the
#   first line.
def getLineData(paper):
    #   Parameters; tweak as needed
    h = 10  # distance from top corner in both dimensions
    segLen = 2
    numLines = 5
    
    #   Similar to technique in getMargin, but vertical. Find the top of the first line.
    start1 = h
    activ = 1000    
    while (activ >= 255*segLen):
        activ = 0
        for i in range(segLen):
            activ += paper.pixels[start1+i][h]
        start1 += segLen  # jump size for start of segment

    #   Find the bottom of the (numLines)th line   
    start2 = start1
    for line in range(numLines):
        #if (line == numLines-1):
            #print("line %s boundary found at %s pixels." %(line, start2))
        start2 += 5     # to avoid detecting the same line
        activ = 1000
        while (activ >= 255*segLen):
            activ = 0
            for i in range(segLen):
                activ += paper.pixels[start2+i][h]
            start2 += segLen  # jump size for start of segment

    lineH = int((start2 - start1)/numLines)

    print("Found the line height! =%s pixels average over %s line(s)." %(lineH,numLines))
    print("Found the top line! =%s pixels from the top.\n" %start1)

    return (lineH, start1)

#   Given the paper and line, returns the y coord of the next region of color
#   "color", where color = 1 corresponds to black (=0 => white). For example,
#   to look for the next whitespace after "start", color = 0. Handles case
#   where "start" = end. A helper function for partitionLine method
def searchLine(paper,line,start,color):
    #   Parameters; tweak as needed
    spacing = 3     # num pixels between line segments
    numSegs = 6
    segWidth = 7
    buff = 4      # num pixels above line to start checking

    #   The trivial case where we've already reached the end of the line
    if start[1] == paper.pixels.shape[1]:
        return paper.pixels.shape[1]

    #   Find next whitespace
    if color == 0:     
        #   Checks horizontal line segments and continues while at least one
        #   occupies a black pixel
        activ = 0
        while (activ < 255*numSegs*segWidth):
            activ = 0        

            #   Identify when line ends
            if (start[1]+segWidth >= paper.pixels.shape[1]):
                print("Detected text to the very end of line %s." %line)
                return paper.pixels.shape[1]
            
            #   Checks line segments
            for seg in range(numSegs):
                for pix in range(segWidth):
                    activ += paper.pixels[start[0]-buff-seg*spacing][start[1]+pix]

            start[1] += segWidth   # to proceed rightward until next color space

        start[1] += 2-segWidth   # to ensure letter is not truncated

    #   Find next word
    else:
        #   Checks horizontal line segments and continues while all are white
        activ = 255*numSegs*segWidth
        while (activ == 255*numSegs*segWidth):
            activ = 0
            
            #   Identify when line ends
            if (start[1]+segWidth >= paper.pixels.shape[1]):
                return paper.pixels.shape[1]
            
            #   Checks line segments
            for seg in range(numSegs):
                for pix in range(segWidth):
                    activ += paper.pixels[start[0]-buff-seg*spacing][start[1]+pix]
                    
            start[1] += segWidth   # to proceed rightward until next color space

        #   For precise word boundaries: since a black pixel was found somewhere,
        #   See if it was the left or right half of the segments that were checked.
        start[1] -= segWidth    # go back to last segment
        activ = 0
        for seg in range(numSegs):
            for pix in range(int(segWidth/2)):
                activ += paper.pixels[start[0]-buff-seg*spacing][start[1]+pix]
        #   If the left half is all white, it must be the right half; may add back
        #   half of a segment length
        if (activ == 255*numSegs*int(segWidth/2)):
            start[1] += int(segWidth/2)

    return start[1]

#   Converts a segment of the paper to a 2d numpy array of the correct
#   dimensions for input to the neural network. The segment is passed by
#   starting and ending w and h coordinates containing a potential letter.
def toImage(w0,h0,w1,h1,paper):
    netWidth = 28
    netHeight = 28
    tol = 200 # threshold brightness above which all pixels are considered white

    #   Want to maintain aspect ratio: determine a constant scaling
    #   factor so that one dimension is 28 and the other is <= 28.
    wRatio = netWidth / (w1-w0)
    hRatio = netHeight / (h0-h1)
    if (wRatio > hRatio):
        narrow = True
    else:
        narrow = False
    wNew = int(round(min(wRatio, hRatio)*(w1-w0)))
    hNew = int(round(min(wRatio, hRatio)*(h0-h1)))

    #   Expand image w/ scaling factor and bicubic interpolation
    im = paper.image
    im = im.resize((wNew,hNew), Image.BICUBIC, (w0,h1,w1,h0)) # PIL uses (width, height)
    startImage = np.asarray(im.getdata(),dtype=np.int16).reshape((im.size[1],im.size[0]))
    #   To black and white
    for row in range(startImage.shape[0]):
        for col in range(startImage.shape[1]):
            if startImage[row][col] > tol:
                startImage[row][col] = 255
            else:
                startImage[row][col] = 0
                
    #print("Converted image (before whitespace) has shape: ", startImage.shape)

    #   Fill in white space if image is too narrow or short
    endImage = np.full((netHeight,netWidth), 255) # All white
    if narrow:
        #   "superimpose" startImage onto the center (laterally) of endImage (or 1 pixel left of center)
        buff = int((netWidth - startImage.shape[1]) / 2)
        #print("Too narrow: trying to add %s pixels on either side." %buff)
        for row in range(netHeight):
            for col in range(buff,netWidth-buff-1):
                endImage[row][col] = startImage[row][col-buff]
    else:
        #   "superimpose" startImage onto the center (vertically) of endImage (or 1 pixel above center)
        buff = int((netHeight - startImage.shape[0]) / 2)
        #print("Too short: trying to add %s pixels above and below." %buff)
        for row in range(buff,netHeight-buff-1):
            for col in range(netWidth):
                endImage[row][col] = startImage[row-buff][col]

    if (endImage.shape[0] != netHeight or endImage.shape[1] != netWidth):
        print("Warning: toImage function did not return proper dimensions!")

    newIm = endImage.flatten()
    #   Invert since 0=white for training data
    for i in range(len(newIm)):
        newIm[i] = 255 - newIm[i]
    return newIm.tolist()

#   Open image and convert to a 2d numpy array of grayscale values. Then return
def getImage(fileName):
    tol = 200   # threshold RGB value to convert pixels to white vs. black
    im = Image.open(fileName, 'r')
    im = im.convert('L',dither=None)    # to grayscale
    pixels = np.asarray(im.getdata(),dtype=np.int16).reshape((im.size[1],im.size[0]))
    #   To black and white
    for row in range(pixels.shape[0]):
        for col in range(pixels.shape[1]):
            if pixels[row][col] > tol:
                pixels[row][col] = 255
            else:
                pixels[row][col] = 0

    print("Opened image! It's %s by %s." %(pixels.shape[0],pixels.shape[1]))
    im.show()
    return (pixels, im)       
