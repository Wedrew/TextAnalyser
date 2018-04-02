import matplotlib.pyplot
import numpy

train = numpy.loadtxt("/Users/andrewpagan/Documents/School/TextAnalyser/data/train-pre3.csv.gz", dtype=numpy.uint8,
                      delimiter=",", skiprows=1)
train_im = train[:, 1:]
train_lab = train[:, 0]
print(train_im.shape)
del train

# In case the images have been flattened into vectors, convert back
# to a two-dimensional image.
for x in train_im:
	im = x.reshape(54, 32)
	matplotlib.pyplot.imshow(im, cmap="gray")
	matplotlib.pyplot.show()