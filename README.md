# TextAnalyser
Utilizes a forward feed neural network to predict text for 28x28 images. Allows training, saving, running, and loading networks in .npy format, testing data must be in csv format. Normally achieves around 97% accuracy with good constraints

# Example Screenshots:
![ScreenShot](https://i.imgur.com/9m7dtNu.png)

# Compiling
Linux:
  * Python3.6, numpy, matplotlib, scipy
  * Clone repository and cd to root directory
  * ```$ unzip data.zip```
  * ```$ python3.6 main.py```

# Training data:
  - https://www.seehuhn.de/maths/letters/index.html
  - https://www.kaggle.com/crawford/emnist/version/3
  - https://www.nist.gov/itl/iad/image-group/emnist-dataset
  - http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
  - http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
