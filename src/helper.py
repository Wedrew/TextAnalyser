import os
import sys

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
    print("******************************")
    print("Available files: ")
    for root, dirs, files in os.walk(rootDir):  
        for filename in files:
            if not filename.startswith('.'):
                print("-" + filename)
    print("******************************")

def printFolders(rootDir):
    print("******************************")
    print("Available folders: ")
    for folder in os.listdir(rootDir):
        print(folder + "/")
    print("******************************")


def loadFile(rootDir):
    pass

def getInput(message, dtype=None):
    while True:
        if dtype == "i":
            try:
                userInput = int(input(message))
                assert userInput > 0
                break
            except (ValueError, AssertionError): 
                print("Try again")
            except KeyboardInterrupt:
                print("Shutdown requested...exiting.")
                sys.exit(0)
        elif dtype == "f":
            try:
                userInput = float(input(message))
                assert userInput > 0.0
                break
            except (ValueError, AssertionError):
                print("Try again")
        else:
            try:
                userInput = input(message)
                assert userInput != " "
                break
            except (ValueError, AssertionError):
                print("Try again")
    return userInput