from PIL import Image
import numpy as np

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
        bot = self.lineH * line + self.topLine
        top = self.lineH * (line-1) + self.topLine
        start = word[0]
        finalWord = ""

        while (start < word[1]):
            maxConf = 0     # largest confidence of the NN for any end
            maxEnd = -1    # end of letter that maximizes NN confidence
            maxLetter = "#" # corresponding letter the NN sees
            maxH = bot-3   # smallest height containing entire current letter

            #   Try all widths of current letter "box" possible up to 28
            for end in range(max(word[1] - start + 1, 28)):
                #   At end pixel laterally, check if letter needs more space by
                #   moving downward until we hit a black pixel
                pix = 255
                currH = top+2   # makes sure top line doesn't mess with loop
                while pix == 255 and currH <= bot-3:
                    #print(currH)
                    currH += 2
                    pix = self.pixels[currH][start+end]
                currH -= 2
                #   If letter is taller than in the other pixel columns so far,
                #   raise the maxH to ensure it's contained in our final "box"
                if currH < maxH:
                    maxH = currH
                    #print("Raising box height to %s" %(bot-maxH))

                #   Letter won't be smaller than 3 pixels wide or tall
                if end >= 3 and bot-maxH >= 3:
                    #   Convert our "box" into an image containing the potential letter
                    #print("Passing the parameters:" , start, bot, start+end, maxH)
                    letterI = toImage(start,bot,start+end,maxH,self)

                    #   Determine confidence and save the end of the box and corresponding
                    #   confidence if it exceeds the previous max. Add the letter the NN
                    #   thinks it is, in this case, to finalWord
                    confAndLetter = network.testLetter(letterI)
                    currConf = confAndLetter[0]
                    letter = str(confAndLetter[1])
                    if currConf > maxConf:
                        maxConf = currConf
                        maxEnd = start+end
                        maxLetter = letter
                #if maxH-bot < 3:
                    #print("Warning: skipping the procesing of a letter that is too short.")

            finalWord += maxLetter
            #print("Found letter", maxLetter)
            start = maxEnd

        #print("Read the word" , finalWord)
        return finalWord
            
# -------------------------------------------------------------------------------------
#
#   Helper functions ( not methods )
#
# -------------------------------------------------------------------------------------


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
        if (line == numLines-1):
            print("line %s boundary found at %s pixels." %(line, start2))
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
        start[1] -= segWidth # to ensure words are not truncated at the left

    return start[1]

#   Converts a segment of the paper to a 2d numpy array of the correct
#   dimensions for input to the neural network. The segment is passed by
#   starting and ending w and h coordinates containing a potential letter.
def toImage(w0,h0,w1,h1,paper):
    netWidth = 28
    netHeight = 28

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

    #   Expand image w/ scaling factor and bilinear interpolation
    im = paper.image.convert('L')
    im = im.resize((wNew,hNew), Image.BILINEAR, (w0,h1,w1,h0)) # PIL uses (width, height)
    startImage = np.asarray(im.getdata(),dtype=np.int16).reshape((im.size[1],im.size[0]))
    #print("Converted image (before whitespace) has shape: ", startImage.shape)

    #   Fill in white space if image is too narrow or short
    endImage = np.full((netHeight,netWidth), 255) # All white
    if narrow:
        #   "superimpose" startImage onto the center (laterally) of endImage (or 1 pixel left of center)
        buff = int((netWidth - startImage.shape[1]) / 2)
        #print("Too narrow: trying to add %s pixels on either side." %buff)
        for row in range(netHeight):
            for col in range(buff,netWidth-buff-1):
                aosdiaufh = startImage[row][col-buff]
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
    #else:
        #print("Successfully produced image for input to NN!")
        
    return list(endImage.flatten())

#   Open image and convert to a 2d numpy array of grayscale values. Then return
def getImage(fileName):
    tol = 200   # threshold RGB value to convert pixels to white vs. black
    im = Image.open(fileName, 'r')
    im = im.convert('L',dither=None)    # to grayscale
    pixels = np.asarray(im.getdata(),dtype=np.int32).reshape((im.size[1],im.size[0]))
    #   To black and white
    for row in range(pixels[0].size):
        for col in range(pixels[1].size):
            if pixels[row][col] > tol:
                pixels[row][col] = 255
            else:
                pixels[row][col] = 0

    print("Opened image! It's %s by %s." %(pixels.shape[0],pixels.shape[1]))
    im.show()
    return (pixels, im)       
