

import sys
# main import: QtWidgets, QtCore, QtGui
from PyQt5 import QtWidgets, QtCore, QtGui
# from PyQt5.QtWidgets import QApplication, QMainWindow

class UImain(object):

    def setupGUI(self,window):
        window.setObjectName("UImain")
        window.resize(400,400)
        window.move(100,100)
        self.centralwidget = QtWidgets.QWidget(window)
        window.setWindowTitle("qtmain window")
        window.setWindowIcon(QtGui.QIcon('setting.png'))



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()

    ui = UImain()
    ui.setupGUI(window)

    window.show()
    sys.exit(app.exec_())

