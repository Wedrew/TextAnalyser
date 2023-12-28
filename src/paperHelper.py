# ----------------------------------------------------------------------
#   paperHelper.py
#
#   Provides all functions needed in the methods defined in paper.py
# ----------------------------------------------------------------------

from PIL import Image
import numpy as np

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

#   Sends the neural network all images it will need in partitionWord, and
#   generates a (maxLen - minLen + 1) length list of lists, where each element
#   is a tuple (letter, confidence) from network.testLetter. The elements
#   of each array are ordered by image starting position from left to right.
def getImageCache(paper, network, minLen, maxLen, word, line):
    #   Define word boundaries
    bot = paper.lineH * (line+1) + paper.topLine
    top = paper.lineH * line + paper.topLine
    start = word[0]
    end = word[1]
    
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
            boxTop = min(tops[startPos+1:startPos+w-1]) # min means highest pixel
            letterI = toImage(start+startPos,bot-1,start+startPos+w,boxTop,paper)
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

#   Open image and convert to a 2d numpy array of black and white values based on a manually
#   set threshold. Returns a tuple with the array and the grayscale (not black/white) Image object
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
    return (pixels, im)       
