import os
import sys

if __name__ == "__main__":
	#Execute application
	print("Running main window")
	app = QApplication(sys.argv)
	main = MainWindow()
	main.show()

	sys.exit(app.exec_())