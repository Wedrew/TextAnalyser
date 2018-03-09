import os

def createFolders(rootDir):
	#Check if savednetworks folder exists (can't upload empty folders to git)
        if not os.path.exists(rootDir + "/savednetworks"):
            #Create folder to hold networks
            os.makedirs(rootDir + "/savednetworks")

        #Check if savednetworks folder exists (can't upload empty folders to git)
        if not os.path.exists(rootDir + "/data"):
            #Create folder to hold networks
            os.makedirs(rootDir + "/data")
            os.makedirs(rootDir + "/data/testing")
            os.makedirs(rootDir + "/data/training")

def printFiles(rootDir):
	for root, dirs, files in os.walk(rootDir + "/data/training/"):  
		for filename in files:
			print("Available training files: ", filename)