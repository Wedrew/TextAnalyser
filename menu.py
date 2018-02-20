import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QInputDialog, QLineEdit, QFileDialog, QDesktopWidget
from PyQt5.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMenu, QMenuBar, QPushButton, QSpinBox, QTextEdit, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

class App(QMainWindow):
	def __init__(self, parent=None):
		super().__init__()
		#Get screen size and set window
		self.title = "Text Analyser"
		resolution = QDesktopWidget().screenGeometry()
		self.left = (resolution.width() / 2) - (self.frameSize().width() / 2)
		self.top = (resolution.height() / 2) - (self.frameSize().height() / 2)
		self.width = 640
		self.height = 480
		self.input_nodes = 0
		self.hidden_nodes = 0
		self.output_nodes = 0
		self.learningrate = 0
		self.initUI()

	def initUI(self):
		#Boilerplate code
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)
		self.loadMenu()
		self.show()

	def loadMenu(self):
		#Initialize menu items here
		mainMenu = self.menuBar()
		mainMenu.setNativeMenuBar(False)
		fileMenu = mainMenu.addMenu("File")
		editMenu = mainMenu.addMenu("Edit")
		helpMenu = mainMenu.addMenu("Help")

		loadButton = QAction("Load", self)
		loadButton.setShortcut("Ctrl+L")
		loadButton.setStatusTip("Load Network")
		loadButton.triggered.connect(self.loadNetwork)

		trainButton = QAction("Train", self)
		trainButton.setShortcut("Ctrl+T")
		trainButton.setStatusTip("Train Network")
		trainButton.triggered.connect(self.trainNetwork)

		quitButton = QAction("Quit", self)
		quitButton.setShortcut("Ctrl+Q")
		quitButton.setStatusTip("Quit Application")
		quitButton.triggered.connect(self.close)

		fileMenu.addAction(trainButton)
		fileMenu.addAction(loadButton)
		fileMenu.addAction(quitButton)

	def loadNetwork(self):
		#Find file name of network user intended to load
		print("Loading file browser")
		options = QFileDialog.Options()
		options = QFileDialog.ShowDirsOnly
		options |= QFileDialog.DontUseNativeDialog
		#self.networkName, _ = QFileDialog.getOpenFileName(self,"File Broswer", "","All Files (*);;Python Files (*.py)", options=options)
		self.networkName = QFileDialog.getExistingDirectory(self, "Select Network", os.getcwd(), options=options)
		print(self.networkName)

		#Attempt try catch for files in folder
		#
		#
		#
		#

	def trainNetwork(self):
		#Loads Dialog class which inherits from
		dialog = Dialog(self)
		dialog.show()

		
class Dialog(QDialog, App):
	def __init__(self, parent=None):
		QDialog.__init__(self, parent)
		self.createFormGroupBox()

		buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
		buttonBox.accepted.connect(self.accept)
		buttonBox.rejected.connect(self.reject)

		mainLayout = QVBoxLayout()
		mainLayout.addWidget(self.formGroupBox)
		mainLayout.addWidget(buttonBox)
		self.setLayout(mainLayout)
		self.setWindowTitle("Network attributes")

	def createFormGroupBox(self):
		self.formGroupBox = QGroupBox("Settings")
		layout = QFormLayout()
		layout.addRow(QLabel("Number of input neurons:"), QLineEdit())
		layout.addRow(QLabel("Number of hidden neurons:"), QLineEdit())
		layout.addRow(QLabel("Number of output neurons:"), QLineEdit())
		layout.addRow(QLabel("Learning rate:"), QLineEdit())
		layout.addRow(QLabel("Amount of epochs:"), QLineEdit())

		self.formGroupBox.setLayout(layout)

if __name__ == "__main__":
	#Execute application
	app = QApplication(sys.argv)
	ex = App()
	print("Running application...")
	sys.exit(app.exec_())