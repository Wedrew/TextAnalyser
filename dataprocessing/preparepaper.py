import cv2
import numpy as np
import PIL
from scipy import misc
import time

#Use misc.imread("foo.jpg")
def removeLines(imageArray):
	image = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

	_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
	connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
	# using RETR_EXTERNAL instead of RETR_CCOMP
	_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	mask = np.zeros(bw.shape, dtype=np.uint8)

	for idx in range(len(contours)):
	    x, y, w, h = cv2.boundingRect(contours[idx])
	    mask[y:y+h, x:x+w] = 0
	    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
	    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

	    #We should calculated row heights (if any exist) before setting limits for boxes
	    if r > 0.2 and w > 5 and h > 10:
	        cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (0, 255, 0), 1)

	data = np.asarray(image, dtype="int32")
	return data


def getBorders(imageArray):
	#Clean up image for processing
	image = cv2.fastNlMeansDenoisingColored(imageArray, None, 10, 10, 7, 21)
	#Convert image to grayscale
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#Coordinates to hold bonder regions
	borderCoordinates = []
	#Create long line kernel, and do morph-close-op
	kernel = np.ones((1,40), np.uint8)
	morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
	#Create long line kernel, and do morph-close-op
	kernel2 = np.ones((40,1), np.uint8)
	morphed2 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel2)
	#Combined lines that need to be removed (assume paper is white)
	totalLines = cv2.add((255-morphed2), (255-morphed))
	#Remove lines from original image
	finalImage = cv2.add(totalLines, image)

	#At this point image should have no paper lines
	# threshold the image, setting all foreground pixels to
	# 255 and all background pixels to 0

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	grad = cv2.morphologyEx(finalImage, cv2.MORPH_GRADIENT, kernel)

	_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
	connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
	# using RETR_EXTERNAL instead of RETR_CCOMP
	_, contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	mask = np.zeros(bw.shape, dtype=np.uint8)

	for idx in range(len(contours)):
	    x, y, w, h = cv2.boundingRect(contours[idx])
	    mask[y:y+h, x:x+w] = 0
	    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
	    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

	    #We should calculated row heights (if any exist) before setting limits for boxes
	    if r > 0.25 and w > 5 and h > 8:
	        cv2.rectangle(finalImage, (x, y), (x+w-1, y+h-1), (0, 255, 0), 1)
	        #borderCoordinates.append([(x,y), (x+w-1, y+h-1)])


	data = np.asarray(finalImage)
	return data

# imageArray = misc.imread("/Users/andrewpagan/Documents/School/TextAnalyser/data/images/testPaper2.jpg")
# convertedImage = getBorders(imageArray)
# im = PIL.Image.fromarray(convertedImage)
# im.show()