from PIL import Image

#Utility to find all images within a folder and combine them into one csv file
#Searches every subdirectory of the folder you give
#Example ---> python buildimagedata.py -p PathToYourFolderWithImages -s FolderToSaveSvgTo
def createCsv(loadPath, savePath):


image = Image.open("5A_00006.png")
pixelData = image.load()
width, height = image.size

#Create training data and write values from image
with open('training_data.csv', 'w+') as f:
	f.write("a,")

	for x in range(width):
		for y in range(height):
			r = pixelData[x,y][0]
			b = pixelData[x,y][1]
			g = pixelData[x,y][2]
			#Calculate grey scale from rgb values
			greyScale = int((r+b+g)/3)
			f.write("{},".format(greyScale))