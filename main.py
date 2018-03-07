from sys import argv
from menu import loadMenu

def getopts(argv):
    opts = {}
    while argv:
        if argv[0][0] == '-':
            opts[argv[0]] = argv[1]
        argv = argv[1:]
    return opts

if __name__ == '__main__':
	myargs = getopts(argv)
	if '-p' in myargs and '-s' in myargs:
		print(myargs['-p'])
		print(myargs['-s'])
		rootDir = myargs['-p']
		saveSir = myargs['-s']
		#Call build image data

	elif not myargs:
		#Load menu function
		loadMenu()

	else:
		print("Incorrect arguments")
		break