from sys import argv
from src.menu import loadMenu
from src.helper import *
import os

if __name__ == '__main__':
	#Get working directory
	rootDir = os.getcwd()
	#Create necessary folders
	createFolders(rootDir)
	loadMenu(rootDir)