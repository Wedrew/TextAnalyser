from PIL import Image
import numpy as np

#   This file contains most code sufficient to open a specified image file
#   and write the contents into a .txt file. Relies on a neural network in
#   a separate file.

#   A class representing a scanned notebook paper containing pixel data,
#   margin width, starting line height, line spacing, and number of lines
class Paper(object):
    def __init__(self):
        self.pixels, self.image = getImage('testPaper.jpg')
        self.margin = getMargin(self)
        self.lineH, self.topLine = getLineData(self)
        self.numLines = (int)((self.pixels.shape[0] - self.topLine) / self.lineH)

    #   Given the paper, returns a list of length 2 arrays [start, end] denoting
    #   the "y" coordinates of the beginning and end of detected words in a line
    #   on the paper. The line number is passed as a parameter.
    def partitionLine(self, line):
        yCoords = []
        tol = 3         # num pixels right of margin

        #   Contains (x,y) coords of current position in line
        start = [self.topLine + line * self.lineH - spacing, self.margin + tol]
        
        end = shape(self.pixels)[1]     # end of line
        #   Until the end of the line, check for next word then next whitespace
        #   Add the y coords of the starts of new words/ whitespaces
        while (start[1] < end):
            start[1] = searchLine(self,line,start,1)
            if start[1] != end:
                yCoords.append(start[1])
            else:
                break
            start[1] = searchLine(self,line,start,0)
            if start[1] != end:
                yCoords.append(start[1])

        #   For nonempty lines, "yCoords" should have odd length iff the line ends
        #   with a word (no extra space) and even length iff there's extra whitespace.
        #   Adds end bound to the last word in the former case
        if len(yCoords) == 0:
            return []
        elif len(yCoords)%2 == 1:
            yCoords.append(end)

        #   Formats yCoords to have desired [start,end] structure for words in line
        words = []
        for i in range(len(yCoords)-1):
            words.append([yCoords[i],yCoords[i+1]])
        print("Successfully read line %s; found %s words," %(line,len(words)))

        return words

    #   Given a word, the paper, and current line, returns the word as a single string
    def partitionWord(self, line, word):
        bot = self.lineH * line + self.topLine
        top = self.lineH * (line-1) + self.topLine
        start = word[0]
        finalWord = ""

        while (start < word[1]):
            maxConf = 0     # largest confidence of the NN for any end
            maxEnd = -1    # end of letter that maximizes NN confidence
            maxLetter = "#" # corresponding letter the NN sees
            maxH = bot   # largest height containing entire current letter

            #   Try all widths of current letter "box" possible
            for end in range(word[1] - start + 1):
                #   At end pixel laterally, check if letter needs more space by
                #   moving downward until we hit a black pixel
                pix = 255
                currH = top
                while pix == 255:
                    currH += 2
                    pix = self.pixels[currH][endPix]
                #   If letter is taller than in the other pixel columns so far,
                #   raise the maxH to ensure it's contained in our final "box"
                if currH < maxH:
                    maxH = currH

                #   Convert our "box" into an image containing the potential letter
                letterI = toImage(start,bot,end,maxH,self)

                #   Determine confidence and save the end of the box and corresponding
                #   confidence if it exceeds the previous max. Add the letter the NN
                #   thinks it is, in this case, to finalWord
                confAndLetter = getConfidence(letterI)
                currConf = confAndLetter[0]
                letter = confAndLetter[1]
                if currConf > maxConf:
                    maxConf = currConf
                    maxEnd = end
                    maxLetter = letter

            finalWord += maxLetter
            start = maxEnd

        print("Read the word" , finalWord)
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
            activ += paper.pixels[start][start+i]
        start += 3  # jump size for start of segment

    print("Found the margin! ~%s pixels from the left." %(start-8))
    return start

#   Given the Paper, and assuming the upper and lower bounds of the first line
#   are clear or marked just to the left of the margin, returns a tuple:
#   (lineH, topL) where lineH is the distance between lines vertically on the
#   Paper, and topL is the distance from the top of the paper to the top of the
#   first line.
def getLineData(paper):
    h = 10  # distance from top corner in both dimensions
    
    #   Similar to technique in getMargin, but vertical. Find the top of the first line.
    start1 = h
    activ = 1000    
    while (activ >= 765): # 765 = 255*3
        activ = 0
        for i in range(3):
            activ += paper.pixels[start1+i][start1]
        start1 += 1  # jump size for start of segment

    #   Find the bottom of the first line   
    start2 = start1 + 5     # to avoid detecting the same line
    activ = 1000
    while (activ > 764):
        activ = 0
        for i in range(3):
            activ += paper.pixels[start2+i][start2]
        start2 += 1  # jump size for start of segment

    lineH = start2 - start1
    print("Found the line height! =%s pixels." %lineH)
    return (lineH, start1)

#   Given the paper and line, returns the y coord of the next region of color
#   "color", where color = 1 corresponds to black (=0 => white). For example,
#   to look for the next whitespace after "start", color = 0. Handles case
#   where "start" = end. A helper function for partitionLine method
def searchLine(paper,line,start,color):
    spacing = 5     # num pixels above bottom of line to start from

    #   The trivial case where we've already reached the end of the line
    if start[1] == shape(paper.pixels)[1]:
        return shape(paper.pixels)[1]

    #   Find next whitespace
    if color == 0:     
        #   Checks 3 horizontal line segments and continues while at least one
        #   occupies a black pixel
        trial = 0
        activ = 0
        while (activ < 255*5):
            activ = 0
            start[1] += trial * 5   # to proceed rightward until next color space

            #   Identify when line ends
            if (start[1]+5 >= shape(paper.pixels)[1]):
                print("Detected text to the very end of line %s." %line)
                return shape(paper.pixels)[1]
            
            #   Checks line segments
            for seg in range(3):
                for pix in range(5):
                    activ += paper.pixels[start[0]-seg*spacing][start[1]+pix]
                    
        start[1] += 5   # to ensure letter is not truncated

    #   Find next word
    else:
        #   Checks 3 horizontal line segments and continues while all are white
        trial = 0
        activ = 255*5
        while (activ == 255*5):
            activ = 0
            start[1] += trial * 5   # to proceed rightward until next color space
            
            #   Identify when line ends
            if (start[1]+5 >= shape(paper.pixels)[1]):
                return shape(paper.pixels)[1]
            
            #   Checks line segments
            for seg in range(3):
                for pix in range(5):
                    activ += paper.pixels[start[0]-seg*spacing][start[1]+pix]

    return start[1]

#   Converts a segment of the paper to a 2d numpy array of the correct
#   dimensions for input to the neural network. The segment is passed by
#   starting and ending w and h coordinates containing a potential letter.
def toImage(w0,h0,w1,h1,paper):
    netWidth = 28
    netHeight = 28
    #   Want to maintain aspect ratio: determine a constant expansion
    #   factor "expFactor" to scale both width and height by
    if (netWidth / (w1-w0) > netHeight / (h1-h0)):
        expFactor = netHeight / (h1-h0)
        narrow = True
    else:
        expFactor = netWidth / (w1-w0)
        narrow = False
    if expFactor < 1:
        print("Warning: attempting to scale down letter instead of expand.")
        
    #   For compactness
    w = int((w1-w0) * expFactor)
    h = int((h1-h0) * expFactor)
    #   Expand image w/ constant aspect ratio until either dimension
    #   meets the dimensions the NN requires (bilinear interpolation)
    im = paper.image.convert('L')
    im = im.resize((w,h), Image.BILINEAR, (w0,h0,w1,h1))
    startImage = np.asarray(im.getdata(),dtype=np.int16).reshape((im.size[1],im.size[0]))

    #   Fill in white space if image is too narrow or short
    endImage = np.full((netHeight,netWidth), 255) # All white
    if narrow:
        #   "superimpose" startImage onto the center (laterally) of endImage
        buffer = (netWidth - startImage.size[1]) / 2
        for row in range(netHeight):
            for col in range(buffer,netWidth-buffer):
                endImage[row][col] = startImage[row][col]
    else:
        #   "superimpose" startImage onto the center (vertically) of endImage
        buffer = (netHeight - startImage.size[0]) / 2
        for row in range(buffer,netHeight-buffer):
            for col in range(netWidth):
                endImage[row][col] = startImage[row][col]

    if (endImage.shape[0] != netHeight or endImage.shape[1] != netWidth):
        print("Warning: toImage function did not return proper dimensions!")
    return endImage

#   Open image and convert to a 2d numpy array of grayscale values. Then return
def getImage(fileName):
    tol = 200   # threshold RGB value to convert pixels to white vs. black
    im = Image.open(fileName, 'r')
    im = im.convert('L', dither=None)    # to grayscale
    width, height = im.size()
    im2 = im.load()
    pixels = np.asarray(im.getdata(),dtype=np.int16).reshape((im.size[1],im.size[0]))
    #   To black and white
    for row in pixels:
        for pixel in row:
            if pixel > tol:
                pixel = 255
            else:
                pixel = 0
    

    print("Opened image! It's %s by %s." %(pixels.shape[0],pixels.shape[1]))
    im.show()
    return (pixels, im)       
