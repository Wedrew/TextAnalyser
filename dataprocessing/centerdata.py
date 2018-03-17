from PIL import Image, ImageFilter, ImageDraw, ImageOps
import PIL.ImageOps
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

#Open Image
im = Image.open("/Users/andrewpagan/Documents/School/TextAnalyser/data/images/long.png")
pix = np.asarray(im)

pix = pix[:,:,0:3] # Drop the alpha channel
idx = np.where(pix-255)[0:2] # Drop the color when finding edges
box = list(map(min,idx))[::-1] + list(map(max,idx))[::-1]

#Crop and apply as numpy array
region = im.crop(box)
region_pix = np.asarray(region)

img = Image.fromarray(region_pix, 'RGB')
img = img.convert("L")
pixels = list(img.getdata())
width, height = img.size

if width == height:
    #Scale
    pass
elif width > height:
    #Will add one tenth the squares value as padding to longest side
    ratio = int(width/10)
    im2 = Image.new("L", (width, width), 255)
    im2.paste(img, (0, int(width/2)-int(height/2)))
    img2 = ImageOps.expand(im2, border=ratio,fill='white')
    img2.save("/Users/andrewpagan/Documents/School/TextAnalyser/data/images/asdf.png")
elif height > width:
    #Will add one tenth the squares value as padding to longest side
    ratio = int(height/10)
    im2 = Image.new("L", (height, height), 255)
    im2.paste(img, (int(height/2)-int(width/2), 0))
    img2 = ImageOps.expand(im2, border=ratio,fill='white')
    img2.save("/Users/andrewpagan/Documents/School/TextAnalyser/data/images/asdf.png")

# subplot(121)
# imshow(pix)
# subplot(122)
# imshow(region_pix)
# show()