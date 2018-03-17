import cv2
import numpy as np
import time
import os

rootDir = os.getcwd()
fileToConvert = rootDir + "/data/images/testPaper2.jpg"
image = cv2.imread(fileToConvert, cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', image)
cv2.waitKey(0)

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

    if r > 0.3 and w > 6 and h > 6:
        cv2.rectangle(image, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

cv2.imshow('rects', image)
cv2.waitKey(0)
# write original image with added contours to disk  
cv2.imwrite(rootDir + "/data/images/testPaperC.jpg", image) 