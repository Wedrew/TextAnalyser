# TextAnalyser
Utilizes a forward feed neural network, written purely in python, to predict hand written text. Allows training, saving, running, drawing letters, and loading networks in .npy format, testing data must be in csv format. Normally achieves around 97% accuracy with good constraints

# Example Screenshots:
<img src="/data/images/2018-05-02 15.55.18.gif?raw=true">

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
