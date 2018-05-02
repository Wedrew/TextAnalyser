<p align="center"><b>We are currently working on porting this application to c/c++ using OpenCV</b><br></p>

# TextAnalyser
Utilizes a forward feed neural network, written purely in python, to predict hand written text. Allows training, saving, running, drawing letters, and loading networks in .npy format, testing data must be in csv format. Normally achieves around 97% accuracy with good constraints. When drawing the network only see's the final image and does not receive any mouse strokes. Also includes algorithms to located the words in lined paper (Stored in /datapreprocessing). If you have any questions please feel free to email me.

# Examples:
<img src="/data/images/network.gif?raw=true">

![ScreenShot](https://i.imgur.com/9m7dtNu.png)

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
