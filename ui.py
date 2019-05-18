import PyQt5

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QGridLayout,QTabWidget, QWidget, QCheckBox, QLineEdit, QLabel, QListView, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, qApp, QAction
from PyQt5 import QtCore, QtGui, uic
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QStandardItem, QFont, QStandardItemModel, QPixmap, QIcon, QTextCursor, QTextImageFormat
from PyQt5.QtCore import QTimer, QSize, Qt

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import sys
import graf

class Main(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.grid  = QtWidgets.QGridLayout(self)
        self.font = QFont("Times", 11)

        self.list = QtWidgets.QListWidget(self)
        self.list.setFont(self.font)

        self.text = QtWidgets.QTextEdit(self)
        self.text.setFixedSize(350, 50)
        self.text.setFont(self.font)

        self.send = QtWidgets.QPushButton(self)
        self.send.clicked.connect(self.send_click)
        self.send.setFont(self.font)
        self.send.setIcon(QIcon('send_message'))

        self.grid.addWidget(self.list, 0, 1, 1, 2)
        self.grid.addWidget(self.text, 1, 1)
        self.grid.addWidget(self.send, 1, 2)

        self.setGeometry(300, 300, 500, 700)
        self.setWindowTitle('Ебать спасибо нахуй')

        self.show()

    def send_click(self):
        self.rhase_text = QListWidgetItem(self.text.toPlainText())
        self.rhase_text.setTextAlignment(Qt.AlignRight)
        self.list.addItem(self.rhase_text)

        otvet_text = graf.screach_claster(self.text.toPlainText())
        self.otvet_rhase_text = QListWidgetItem(str(otvet_text))
        self.otvet_rhase_text.setTextAlignment(Qt.AlignLeft)
        self.list.addItem(self.otvet_rhase_text)
        #self.list.addItem(str(otvet_text))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(open("./style.qss", "r").read())
    main = Main()
    sys.exit(app.exec_())