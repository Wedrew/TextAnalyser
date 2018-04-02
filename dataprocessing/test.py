from skimage import img_as_bool, io, color, morphology
import matplotlib.pyplot as plt
import numpy as np
import cv2
emnistLetterMapping = {
	0:'A',
	1:'B',
	2:'C',
	3:'D',
	4:'E',
	5:'F',
	6:'G',
	7:'H',
	8:'I',
	9:'J',
	10:'K',
	11:'L',
	12:'M',
	13:'N',
	14:'O',
	15:'P',
	16:'Q',
	17:'R',
	18:'S',
	19:'T',
	20:'U',
	21:'V',
	22:'W',
	23:'X',
	24:'Y',
	25:'Z',
}

fileName = str(input("File name: "))
filePath = "/Users/andrewpagan/Documents/School/TextAnalyser/data/training/" + fileName

with open(filePath, "r") as file:
	x=1
	for record in file:
		data = record.split(",")
		correctLetter = emnistLetterMapping[int(data[0])]
		print("Correct letter: {}".format(correctLetter))
		del data[0]
		data = np.asarray(data, np.uint8)
		kernel = np.ones((10,10), np.uint8)
		kernel2 = np.ones((5,5), np.uint8)

		imageArray = np.asarray(data).reshape((28,28))
		#imageArray = cv2.resize(imageArray, (560,560), cv2.INTER_NEAREST)
		#imageArray = 255-imageArray
		#imageArray = cv2.dilate(imageArray, kernel2, iterations=5)
		#imageArray = 255-imageArray
	
		image = img_as_bool(color.rgb2gray(imageArray))
		out = morphology.skeletonize(image)

		out = out.astype(np.uint8)*255
		# out = cv2.dilate(out, kernel, iterations=1)
		#out = cv2.GaussianBlur(out,(5,5),0)

		cv2.imshow("skel", 255-out)
		cv2.imshow("original", 255-imageArray)
		cv2.waitKey(0)



