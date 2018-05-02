<p align="center"><b>We are currently working on porting this application to c/c++ using OpenCV</b><br></p>

# TextAnalyser
Utilizes a forward feed neural network, written purely in python, to predict hand written text. Allows training, saving, running, drawing letters, and loading networks in .npy format, testing data must be in csv format. Normally achieves around 97% accuracy with good constraints. When drawing the network only see's the final image and does not receive any mouse strokes. Also includes algorithms to located the words in lined paper (Stored in /datapreprocessing). If you have any questions please feel free to email me.

# Examples:
<img src="/data/images/network.gif?raw=true">

![ScreenShot](https://i.imgur.com/9m7dtNu.png)

# Proposed 8 step pipeline to segment paper

## First Step

<img align="left" width="200" src="https://i.imgur.com/x6tt46N.jpg">


Original Image:
```
First we need to take the picture, who'd have guessed ;)









```

---


## Second Step

<img align="left" width="200" src="https://i.imgur.com/HvWZswJ.png">

Normalized Image:
```
After taking the image the next step is to perform a couple of dilations 
and blurs before passing the image through OpenCV's normalize function. 
Take a look at normalizeImage in preparepaper.py







```


---

## Third Step

<img align="left" width="200" src="https://i.imgur.com/K9jBH5j.png">

Normalized Image Without Lines:
```
For this step we need to create two long line kernels to describe the 
vertical and horizontal lines. Then do a morphological close up to get 
the appropriate areas. After doing so we combine the lines and subtract
them from the normalized image.






```

---

## Fourth Step

<img align="left" width="200" src="https://i.imgur.com/4x0nqw7.png">

Horizontal Lines Only:
```
At this point we have an image that is very clean but still very hard 
to deteremine where the lines should be separated. My original though 
was to use a horizontal histogram of the image and take the derivative 
of the graph to find the areas most likely to be where the lines are. 
This works very well UNLESS the images are curved and you end up with 
imperfect segmentations. (The lines are very often curved). So instead
my method is to first find the horizontal lines in the image using
morphological operations.


```

---


## Fifth Step

<img align="left" width="200" src="https://i.imgur.com/fq0BtGA.png">

Denoised Horizontal Lines:
```
Before we use the horizontal lines to separate the paper we need to
clean them up and get rid of erroneous pixels. We perform a non local
means denoising to help.







```

---


## Sixth Step

<img align="left" width="200" src="https://i.imgur.com/pdnot3s.png">

Dilated Image:
```
Now that we have perform more morphological operations on the image to
enhance the vertical lines, guaranteeing connectivity. First we dilate
in relation to the width of the paper and the assumed width of each lines.







```

---


## Seventh Step

<img align="left" width="200" src="https://i.imgur.com/dpCCh19.png">

Segmented Lines and Words:
```
Almost there! This step is really three combined into one for brevity.
The idea is to dilate the image from third step vertically and store 
it in a buffer. We make the asssumption that letters will be closer 
together than words and by doing this is "lumps" together the words. 
Finally to guarantee non-connectivity of the different lines we subtract 
the lines thatwe determined in the previous step. At this point all 
of the lines and wordsshould not be touching in any way and completely 
free of noise.


```

---


## Eighth Step

<img align="left" width="200" src="https://i.imgur.com/TiZgalM.png">

Final Segmented Image:
```
Finally we use a find contours function to determine where the 
different "lumps" are. Contours can be explained as a curve joining 
all the continuous points and by using this we effectively determine 
the location of each and every word. We have to filter out some 
contours that aren't large or small enough to be considered words. 
And finally we have our segmented paper :) There are some flaws
with this technique as it causes any below the line character to
be interpreted as a letter however I believe this can be fixed with
some heuristic.


```

---

##  Original Image with Segmentations
<img align="left" width="400" src="https://i.imgur.com/8omdE7u.jpg" hspace="20">


# Compiling
Linux:
  * Python3.6, numpy, matplotlib, scipy, PyOpenCl
  * Clone repository and cd to root directory
  * ```$ unzip data.zip```
  * ```$ python3.6 main.py```

# Training data:
  - https://www.seehuhn.de/maths/letters/index.html
  - https://www.kaggle.com/crawford/emnist/version/3
  - https://www.nist.gov/itl/iad/image-group/emnist-dataset
  - http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
  - http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
