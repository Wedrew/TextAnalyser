import sys
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QDesktopWidget, QWidget
from PyQt5.QtGui import QIcon

class mainWindow(QMainWindow):
    
	def __init__(self):
		super().__init__()
        
		self.initUI()
        
        
	def initUI(self):

		exitAct = QAction(QIcon('exit.png'), '&Exit', self)
		exitAct.setShortcut('Ctrl+Q')
		exitAct.setStatusTip('Exit Application')
		exitAct.triggered.connect(qApp.quit)

		self.resize(640, 480)
		self.statusBar().showMessage('Ready')
		self.center()
		self.setWindowTitle('Neural Network') 
		self.statusBar()  

		menubar = self.menuBar()
		menubar.setNativeMenuBar(False)
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(exitAct)

		self.show()

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())
        
if __name__ == '__main__':
    
	app = QApplication(sys.argv)
	ex = mainWindow()
	sys.exit(app.exec_())