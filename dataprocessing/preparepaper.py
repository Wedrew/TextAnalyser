import cv2
import numpy as np
import PIL
from scipy import misc
import math

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
	    if r > 0.25 and w > 5 and h > 10:
	        cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (0, 255, 0), 1)

	return data


def getBorders(imageArray):
	TARGET_PIXEL_AREA = 2073600.0 #Roughly 1920x1080
	mser = cv2.MSER_create()
	#Convert image to grayscale
	originalImage = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
	#Clean up image for processing
	#Scale image if larger than raget pixel area
	while originalImage.shape[0] * originalImage.shape[1] > TARGET_PIXEL_AREA:
		ratio = float(originalImage.shape[1]) / float(originalImage.shape[0])
		originalImage = cv2.resize(originalImage, None, fx=ratio, fy=ratio, interpolation = cv2.INTER_AREA)

	#
	dilatedImage = cv2.dilate(originalImage, np.ones((10,10), np.uint8))
	bgImage = cv2.medianBlur(dilatedImage, 21)
	diffImage = 255 - cv2.absdiff(originalImage, bgImage)
	normImage = diffImage.copy()
	cv2.normalize(diffImage, normImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) 
	_, thrImage = cv2.threshold(normImage, 230, 0, cv2.THRESH_TRUNC)
	cv2.normalize(thrImage, thrImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

	cv2.imshow("asdf", thrImage)
	cv2.waitKey(0)

	#Coordinates to hold bonder regions
	borderCoordinates = []
	#Create long line kernel, and do morph-close-op
	kernel = np.ones((1,40), np.uint8)
	morphed = cv2.morphologyEx(thrImage, cv2.MORPH_CLOSE, kernel)
	#Create long line kernel, and do morph-close-op
	kernel2 = np.ones((40,1), np.uint8)
	morphed2 = cv2.morphologyEx(thrImage, cv2.MORPH_CLOSE, kernel2)
	#Combined lines that need to be removed (assume paper is white)
	totalLines = cv2.add((255-morphed2), (255-morphed))
	#Remove lines from original image
	cleanedImage = cv2.add(totalLines, thrImage)

	cv2.imshow("asdf", cleanedImage)
	cv2.waitKey(0)

	blurredImage = cv2.erode(cleanedImage, np.ones((1, 10)))
	blurredImage = cv2.dilate(blurredImage, np.ones((1, 22)))

	regions = mser.detectRegions(blurredImage)
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
	cv2.polylines(blurredImage, hulls, 1, (0,255,0)) 

	cv2.imshow("asdf", blurredImage)
	cv2.waitKey(0)

	#At this point image should have no paper lines
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	grad = cv2.morphologyEx(cleanedImage, cv2.MORPH_GRADIENT, kernel)

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
		#Another option is to use scaled image based on average word length
		if r > 0.2 and w > 5 and h > 7: # h should be slightly larger than line width
			cv2.rectangle(cleanedImage, (x, y), (x+w-1, y+h-1), (0, 255, 0), 1)
			# croppedWord = cleanedImage[y:y+h-1, x:x+w-1]
			# cv2.imshow("asdf", croppedWord)
			# cv2.waitKey()
			# borderCoordinates.append([(x,y), (x+w-1, y+h-1)])

	return cleanedImage

imageArray = misc.imread("/Users/andrewpagan/Documents/School/TextAnalyser/data/images/paper2.jpg", mode="RGB")
convertedImage = getBorders(imageArray)
im = PIL.Image.fromarray(convertedImage)
im.show()