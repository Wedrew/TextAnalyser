import cv2
import numpy as np
from dataprocessing.centerdata import centerImage

def createDrawWindow(neuralNetwork):
	global drawing
	drawing = False #True if mouse is pressed
	global mode
	mode = True
	global ix, iy#If True, draw rectangle. Press 'm' to toggle to curve
	ix,iy = -1,-1

	#Mouse callback function
	def draw_circle(event,x,y,flags,param):
	    global ix,iy,drawing,mode

	    if event == cv2.EVENT_LBUTTONDOWN:
	        drawing = True
	        ix,iy = x,y

	    elif event == cv2.EVENT_MOUSEMOVE:
	        if drawing == True:
	        	cv2.circle(img,(x,y),15,(0,0,0),-1)

	    elif event == cv2.EVENT_LBUTTONUP:
	        drawing = False
	        cv2.circle(img,(x,y),15,(0,0,0),-1)
	        #Call neural network for letter
	        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	        grayImage = cv2.resize(grayImage, (28,28), interpolation = cv2.INTER_AREA)
	        grayImage = 255-grayImage
	        grayImage = centerImage(grayImage, 5)
	        grayImage = grayImage.flatten()
	        neuralNetwork.testLetter(grayImage)

	img = np.zeros((280,280,3), np.uint8)
	img[img < 255] = 255
	cv2.namedWindow('Letter')
	cv2.setMouseCallback('Letter',draw_circle)

	while(1):
	    cv2.imshow('Letter',img)
	    key = cv2.waitKey(1) & 0xFF
	    if key == ord('c'):
	    	img = np.zeros((280,280,3), np.uint8)
	    	img[img < 255] = 255
	    elif key == 27:
	        break

	cv2.destroyAllWindows()