import cv2
import numpy as np
image = cv2.imread("/Users/andrewpagan/Documents/School/TextAnalyser/data/images/paper1.jpg")


TARGET_PIXEL_AREA = 2073600.0 #Roughly 1920x1080
scaledImage = image
while scaledImage.shape[0] * scaledImage.shape[1] > TARGET_PIXEL_AREA:
    ratio = float(scaledImage.shape[1]) / float(scaledImage.shape[0])
    scaledImage = cv2.resize(scaledImage, None, fx=ratio, fy=ratio, interpolation = cv2.INTER_AREA)

#grayscale
gray = cv2.cvtColor(scaledImage,cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)

_,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thresh = cv2.dilate(thresh,None)

(_,cnts,_) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])
digits = []
boxes = []

for (i,c) in enumerate(cnts):
    if cv2.contourArea(c)<avgCntArea/10:
        continue
    mask = np.zeros(gray.shape,dtype="uint8")
    (x,y,w,h) = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    cv2.drawContours(mask,[hull],-1,255,-1)
    mask = cv2.bitwise_and(thresh,thresh,mask=mask)
    digit = mask[y-3:y+h+3,x-3:x+w+3]
    #digit = cv2.resize(digit,(28,28))
    boxes.append((x,y,w,h))
    cv2.rectangle(scaledImage, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imshow("Aoeu", scaledImage)
    cv2.waitKey(0)
    digits.append(digit)