from PIL import ImageTk, Image, ImageDraw, ImageFilter
import PIL.ImageOps
from tkinter import *
import time
import os

width = 28
height = 28
rootDir = os.getcwd()

def convert():
    filename = "data/images/character.png"
    image2 = image
    image2 = PIL.ImageOps.invert(image2)
    image2 = image2.resize((28, 28), resample=PIL.Image.BICUBIC)
    pixels = list(image2.getdata())
    file = open(rootDir + "/data/images/character.txt","w+")

    y = 1
    for item in pixels:
        if y == len(pixels):
            file.write(str(item))
        else:
            file.write(str(item)+",")
            y += 1
    
    # image.save(filename)
    # image2 = PIL.Image.open(filename, "r")
    # os.remove("data/images/character.png")
    # image2 = PIL.ImageOps.invert(image2)
    # image2 = image2.filter(ImageFilter.BLUR)
    # pixels = list(image2.getdata())
    # file = open(rootDir + "/data/images/character.txt","w+")
    print("File created")
    file.close()

def clear():
    image = PIL.Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    print("asdf")

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_rectangle(x1, y1, x2, y2, fill="black", width=1)
    draw.rectangle([x1, y1, x2, y2],fill="black")
    #draw.line([x1, y1, x2, y2],fill="black",width=1)

root = Tk()

cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

image = PIL.Image.new("L", (28, 28), 255)
draw = ImageDraw.Draw(image)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)

button=Button(text="Convert",command=convert, justify="left")
button2=Button(text="Clear", command=clear, justify="right")
button.pack()
button2.pack()
root.mainloop()