import cv2
import numpy as np
from dataprocessing.centerdata import centerImage
from dataprocessing.unskewdata import unskewImage
from dataprocessing.skeleton import skeletonImage

def createDrawWindow(neuralNetwork):
	global drawing
	drawing = False
	global mode
	mode = True
	global ix, iy
	ix,iy = -1,-1
	global skeleton
	skeleton = False
	global windowSize
	windowSize = 500

	#Mouse callback function
	def drawCircle(event, x, y, flags, param):
		global ix,iy,drawing,mode

		if event == cv2.EVENT_LBUTTONDOWN:
			drawing = True
			ix,iy = x,y

		elif event == cv2.EVENT_MOUSEMOVE:
			if drawing == True:
				cv2.circle(image,(x,y),15,(0,0,0),-1)

		elif event == cv2.EVENT_LBUTTONUP:
			drawing = False
			if skeleton == False:
				cv2.circle(image,(x,y),15,(0,0,0),-1)
				grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				grayImage = 255-grayImage
				grayImage = centerImage(grayImage, 10)
				#grayImage = unskewImage(grayImage, grayImage.shape[0])
				grayImage = cv2.resize(grayImage, (28,28), interpolation = cv2.INTER_AREA)
				cv2.imshow("Network Letter", 255-grayImage)
				grayImage = grayImage.flatten()
				#Call neural network for letter
				neuralNetwork.testLetter(grayImage)
			elif skeleton == True:
				cv2.circle(image,(x,y),15,(0,0,0),-1)
				grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				grayImage = 255-grayImage
				grayImage = centerImage(grayImage, 1)
				grayImage = skeletonImage(grayImage, 10, 28)
				#grayImage = unskewImage(grayImage, grayImage.shape[0])
				cv2.imshow("Network Letter", 255-grayImage)
				grayImage = grayImage.flatten()
				#Call neural network for letter
				neuralNetwork.testLetter(grayImage)


	image = np.zeros((windowSize,windowSize,3), np.uint8)
	image[image < 255] = 255
	cv2.namedWindow("Letter")
	cv2.setMouseCallback("Letter", drawCircle)
	cv2.namedWindow("Network Letter")
	print("s to toggle skeleton image")
	print("c to clear")
	print("esc to quit")

	while(1):
		cv2.imshow('Letter', image)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('c'):
			image = np.zeros((windowSize,windowSize,3), np.uint8)
			image[image < 255] = 255
		elif key == ord('s'):
			skeleton = not skeleton
			print("Skeleton mode activated") if skeleton==True else print("Skeleton mode deactivated")
			image = np.zeros((windowSize,windowSize,3), np.uint8)
			image[image < 255] = 255
		elif key == 27:
			break

	cv2.destroyAllWindows()