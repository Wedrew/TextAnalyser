import cv2
import numpy as np
import PIL
from scipy import misc
import matplotlib.pyplot as plt
import math

def normalizePaper(originalImage):
	dilatedImage = cv2.dilate(originalImage, np.ones((10,10), np.uint8))
	bgImage = cv2.medianBlur(dilatedImage, 21)
	diffImage = 255 - cv2.absdiff(originalImage, bgImage)
	normImage = diffImage.copy()
	cv2.normalize(diffImage, normImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) 
	_, thrImage = cv2.threshold(normImage, 250, 0, cv2.THRESH_TRUNC)
	cv2.normalize(thrImage, thrImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	return thrImage

def normalizeLines(originalImage):
	dilatedImage = cv2.dilate(originalImage, np.ones((10,10), np.uint8))
	bgImage = cv2.medianBlur(dilatedImage, 21)
	diffImage = 255 - cv2.absdiff(originalImage, bgImage)
	normImage = diffImage.copy()
	cv2.normalize(diffImage, normImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) 
	return normImage


def getBorders(imageArray):
	TARGET_PIXEL_AREA = 2073600.0 #Roughly 1920x1080
	mser = cv2.MSER_create()
	#Scale image if larger than raget pixel area
	scaledImage = imageArray
	while scaledImage.shape[0] * scaledImage.shape[1] > TARGET_PIXEL_AREA:
		ratio = float(scaledImage.shape[1]) / float(scaledImage.shape[0])
		scaledImage = cv2.resize(scaledImage, None, fx=ratio, fy=ratio, interpolation = cv2.INTER_AREA)
	#Keep backup image
	backupImage = scaledImage
	#Convert image to grayscale
	grayImage = cv2.cvtColor(scaledImage, cv2.COLOR_BGR2GRAY)

	#Get cleaned image of paper
	nImage = normalizePaper(grayImage)
	# cv2.imshow("Normalize Paper", nImage)
	# cv2.waitKey(0)

	#Create long line kernel, and do morph-close-op
	kernel = np.ones((1,100), np.uint8)
	morphed = cv2.morphologyEx(nImage, cv2.MORPH_CLOSE, kernel)
	#Create long line kernel, and do morph-close-op
	kernel2 = np.ones((100,1), np.uint8)
	morphed2 = cv2.morphologyEx(nImage, cv2.MORPH_CLOSE, kernel2)
	#Combined lines that need to be removed (assume paper is white)
	totalLines = cv2.add((255-morphed2), (255-morphed))
	#Remove lines from original image
	cleanedImage = cv2.add(totalLines, nImage)

	#Clean erroneous 
	cleanedImage = cv2.fastNlMeansDenoising(cleanedImage, None, 30, 3, 21)
	# cv2.imshow("Deniosed", cleanedImage)
	# cv2.waitKey(0)

	#Make almost white pixels completely white
	a = cleanedImage > 235
	cleanedImage[a] = 255
	cv2.imshow("Bitwise not", cleanedImage)
	cv2.waitKey(0)

	rgb_planes = cv2.split(grayImage)
	result_norm_planes = []
	for plane in rgb_planes:
	    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
	    bg_img = cv2.medianBlur(dilated_img, 21)
	    diff_img = 255 - cv2.absdiff(plane, bg_img)
	    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	    result_norm_planes.append(norm_img)

	result_norm = cv2.merge(result_norm_planes)
	result_norm = cv2.add(result_norm, 255-cleanedImage)

	cv2.imshow('shadows_out_norm.png', result_norm)
	cv2.waitKey(0)

	#Get vertical lines for segementation
	kernel = np.ones((1,10), np.uint8)
	morphed = cv2.morphologyEx(result_norm, cv2.MORPH_CLOSE, kernel)
	cv2.imshow("Vertical Lines", morphed)
	cv2.waitKey(0)

	#Inhance vertical lines
	nLinesImage = normalizeLines(morphed)
	_, nLines = cv2.threshold(nLinesImage, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	nLines = cv2.erode(nLines, np.ones((3,30), np.uint8))
	cv2.imshow("Enhanced Vertical Lines", nLines)
	cv2.waitKey(0)

	dilatedImage = cv2.erode(cleanedImage, np.ones((1,5), np.uint8))
	final = cv2.add(dilatedImage, 255-nLines)
	cv2.imshow("Final", final)
	cv2.waitKey(0)


	#At this point image should have no paper lines
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	grad = cv2.morphologyEx(final, cv2.MORPH_GRADIENT, kernel)

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
		if r > 0.0 and w > 10 and h > 10: # h should be slightly larger than line width 
			cv2.rectangle(scaledImage, (x-2, y-5), (x+w+2, y+h+5), (255, 0, 0), 1)
			# croppedWord = cleanedImage[y:y+h+5, x:x+w]
			# cv2.imshow("asdf", croppedWord)
			# cv2.waitKey(0)
	return scaledImage

imageArray = misc.imread("/Users/andrewpagan/Documents/School/TextAnalyser/data/images/paper1.jpg", mode="RGB")
convertedImage = getBorders(imageArray)
im = PIL.Image.fromarray(convertedImage)
im.show()