import sys
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication

from PyQt5.QtWidgets import QSizePolicy

from PyQt5.QtWidgets import QCheckBox    # checkbox
from PyQt5.QtWidgets import QPushButton  # pushbutton
from PyQt5.QtWidgets import QLineEdit    # Lineedit
from PyQt5.QtWidgets import QRadioButton # Radio Buttons
from PyQt5.QtWidgets import QGroupBox    # Group Box


# These components are essential for creating the graphics in pqt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#---------------------------------------------------------------------

from numpy.polynomial.polynomial import polyfit
import numpy as np

#----------------------------------------------------------------------
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt  # Control status
from PyQt5.QtWidgets import  QWidget,QLabel, QVBoxLayout, QHBoxLayout, QGridLayout

#::------------------------------------------------------------------------------------
#:: Class: Graphic with Params
#::------------------------------------------------------------------------------------
class GraphWParamsClass(QMainWindow):
    send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(GraphWParamsClass, self).__init__()

        self.Title = 'Title : Graphic with Parameters '
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QVBoxLayout (Vertical Layout )
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.v = "With Outliers"
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout

        self.groupBox2 = QGroupBox('With or without Outliers')
        self.groupBox2Layout = QHBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.groupBox3 = QGroupBox('Graphic')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        # Radio buttons are create to be added to the second group

        self.b1 = QRadioButton("With Outliers")
        self.b1.setChecked(True)
        self.b1.toggled.connect(self.onClicked)

        self.b2 = QRadioButton("Without Outliers")
        self.b2.toggled.connect(self.onClicked)

        self.buttonlabel = QLabel(self.v+' is selected')

        self.groupBox2Layout.addWidget(self.b1)
        self.groupBox2Layout.addWidget(self.b2)
        self.groupBox2Layout.addWidget(self.buttonlabel)

        # figure and canvas figure to draw the graph is created to
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        # Canvas is added to the third group box
        self.groupBox3Layout.addWidget(self.canvas)

        # Adding to the main layout the groupboxes
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(600, 500)                         # Resize the window
        self.onClicked()


    def onClicked(self):

        # Figure is cleared to create the new graph with the choosen parameters
        self.ax1.clear()

        train = pd.read_csv('../Final Project/train.csv', parse_dates=["first_active_month"])

        # the buttons are inspect to indicate which one is checked.
        # vcolor is assigned the chosen color
        if self.b1.isChecked():
            self.v = self.b1.text()

        if self.b2.isChecked():
            self.v = self.b2.text()

            idx = train[np.abs(stats.zscore(train['target'])) > 3].index
            # Delete these row indexes from dataFrame
            train.drop(idx, inplace=True)

        # the label that displays the selected option
        self.buttonlabel.setText(self.v+' is selected')

        target = train['target']

        self.ax1.hist(target.values, bins=200)

        vtitle = "Histogram Target Counts"
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel('Count')
        self.ax1.set_ylabel('Target')
        self.ax1.grid(True)

        # show the plot
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

class GLayoutclass(QMainWindow):  ## All the class was added in No. 3 Section
    send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(GLayoutclass, self).__init__()

        self.Title = 'Linear Regression'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QGridLayout (Horizontal Layout )
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)   # Creates horizontal layout
        self.label1 = QLabel("best_score (neg_mean_squared_error): -0.90")        # Creates label1
        self.label2 = QLabel("best_params: {'estimator__eta': 0.1}")        # Creates label2
        self.label3 = QLabel("Mean_squared_error: 2.65")        # Creates label3
        self.label4 = QLabel("r2_score: 0.09")        # Creates label4
        self.layout.addWidget(self.label1,0,0)            # Add label 1 to  layout
        self.layout.addWidget(self.label2,0,1)            # Add label 2 to layout
        self.layout.addWidget(self.label3,1,0)            # Add label 3 to layout
        self.layout.addWidget(self.label4,1,1)            # Add label 4 to layout

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(300, 100)                         # Resize the window

class DT(QMainWindow):
    send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(DT, self).__init__()

        self.Title = 'Title : Decision Tree '
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QVBoxLayout (Vertical Layout )
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout

        self.groupBox2 = QGroupBox('Results')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.groupBox3 = QGroupBox('Graphic')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        self.label1 = QLabel("Best Hyperparameter: {'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2}")
        self.label2 = QLabel("Accuracy: 34.33")

        self.groupBox2Layout.addWidget(self.label1)
        self.groupBox2Layout.addWidget(self.label2)
        # figure and canvas figure to draw the graph is created to
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        df_cm = pd.read_csv('dt_cm_6.csv')
        im = self.ax1.imshow(df_cm)

        # We want to show all ticks...
        self.ax1.set_xticks(np.arange(1, df_cm.shape[0]+1))
        self.ax1.set_yticks(np.arange(df_cm.shape[0]))
        # ... and label them with the respective list entries
        self.ax1.set_xticklabels(df_cm.columns[1:])
        self.ax1.set_yticklabels(df_cm.columns[1:])

        # Rotate the tick labels and set their alignment.
        plt.setp(self.ax1.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(df_cm.shape[0]):
            for j in range(1, df_cm.shape[0]+1):
                text = self.ax1.text(j, i, df_cm.iloc[i, j],
                               ha="center", va="center", color="w", fontsize=6)
        vtitle = "Confusion Matrix"
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel('Predicted label', fontsize=10)
        self.ax1.set_ylabel('True label', fontsize=10)

        # Canvas is added to the third group box
        self.groupBox3Layout.addWidget(self.canvas)

        # Adding to the main layout the groupboxes
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(600, 500)                         # Resize the window
        # show the plot
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


class RF(QMainWindow):
    send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(RF, self).__init__()

        self.Title = 'Title : Random Forest '
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QVBoxLayout (Vertical Layout )
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)  # Creates vertical layout

        self.groupBox2 = QGroupBox('Results')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.groupBox3 = QGroupBox('Graphic')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        self.label1 = QLabel("Best Hyperparameter: {'max_depth': 6, 'min_samples_leaf': 1, 'min_samples_split': 2}")
        self.label2 = QLabel("Accuracy: 34.33")

        self.groupBox2Layout.addWidget(self.label1)
        self.groupBox2Layout.addWidget(self.label2)
        # figure and canvas figure to draw the graph is created to
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        df_cm = pd.read_csv('dt_cm_6.csv')
        im = self.ax1.imshow(df_cm)

        # We want to show all ticks...
        self.ax1.set_xticks(np.arange(1, df_cm.shape[0] + 1))
        self.ax1.set_yticks(np.arange(df_cm.shape[0]))
        # ... and label them with the respective list entries
        self.ax1.set_xticklabels(df_cm.columns[1:])
        self.ax1.set_yticklabels(df_cm.columns[1:])

        # Rotate the tick labels and set their alignment.
        plt.setp(self.ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(df_cm.shape[0]):
            for j in range(1, df_cm.shape[0] + 1):
                text = self.ax1.text(j, i, df_cm.iloc[i, j],
                                     ha="center", va="center", color="w", fontsize=6)
        vtitle = "Confusion Matrix"
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel('Predicted label', fontsize=10)
        self.ax1.set_ylabel('True label', fontsize=10)

        # Canvas is added to the third group box
        self.groupBox3Layout.addWidget(self.canvas)

        # Adding to the main layout the groupboxes
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)

        self.setCentralWidget(self.main_widget)  # Creates the window with all the elements
        self.resize(600, 500)  # Resize the window
        # show the plot
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


#::-------------------------------------------------------------
#:: Definition of a Class for the main manu in the application
#::-------------------------------------------------------------
class Menu(QMainWindow):

    def __init__(self):

        super().__init__()
        #::-----------------------
        #:: variables use to set the size of the window that contains the menu
        #::-----------------------
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 300

        #:: Title for the application

        self.Title = 'Elo Merchant Category Recommendation-Help understand customer loyalty'

        #:: The initUi is call to create all the necessary elements for the menu

        self.initUI()

    def initUI(self):

        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()
        #::-----------------------------
        # 1. Create the menu bar
        # 2. Create an item in the menu bar
        # 3. Creaate an action to be executed the option in the  menu bar is choosen
        #::-----------------------------
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')

        #:: Add another option to the Menu Bar

        exampleWin = mainMenu.addMenu ('Graphics')

        #::--------------------------------------
        # Exit action
        # The following code creates the the da Exit Action along
        # with all the characteristics associated with the action
        # The Icon, a shortcut , the status tip that would appear in the window
        # and the action
        #  triggered.connect will indicate what is to be done when the item in
        # the menu is selected
        # These definitions are not available until the button is assigned
        # to the menu
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), '&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        #:: This line adds the button (item element ) to the menu

        fileMenu.addAction(exitButton)

        #::------------------------------------------------------------
        #:: Add code to include Text Line and button to implement an action upon request
        #::------------------------------------------------------------

        #::------------------------------------------------------------
        #:: Add code to include radio buttons  to implement an action upon request
        #::------------------------------------------------------------

        exampleGWParams =  QAction("Histogram Target Count ", self)
        exampleGWParams.setStatusTip('Example of Graphic with parameters')
        exampleGWParams.triggered.connect(self.ExampleGraphWParams)

        exampleWin.addAction(exampleGWParams)

        example4Button = QAction("Linear Regression", self)  # No. 3
        example4Button.setStatusTip("Example of Grid layout")  # No. 3
        example4Button.triggered.connect(self.GLayout)  # No. 3

        #:: We addd the example2Button to the menu examples
        exampleWin.addAction(example4Button)

        exampleGraphic =  QAction("Decision Tree", self)
        exampleGraphic.setStatusTip('Example of Graphic')
        exampleGraphic.triggered.connect(self.DT)

        exampleWin.addAction(exampleGraphic)

        exampleGraphic2 = QAction("Random Forest", self)
        exampleGraphic2.setStatusTip('Example of Graphic')
        exampleGraphic2.triggered.connect(self.RF)

        exampleWin.addAction(exampleGraphic2)

        #:: Creates an empty list of dialogs to keep track of
        #:: all the iterations

        self.dialogs = list()

        #:: This line shows the windows
        self.show()

    def ExampleGraphWParams(self):
        dialog = GraphWParamsClass()
        self.dialogs.append(dialog) # Apppends to the list of dialogs
        dialog.show()

    def GLayout(self):  # No. 3
        dialog = GLayoutclass()    # Creates an object with the Horizontal class
        self.dialogs.append(dialog) # Appeds the list of dialogs
        dialog.show()

    def DT(self):
        dialog = DT()
        self.dialogs.append(dialog) # Apppends to the list of dialogs
        dialog.show()

    def RF(self):
        dialog = RF()
        self.dialogs.append(dialog) # Apppends to the list of dialogs
        dialog.show()
#::------------------------
#:: Application starts here
#::------------------------

def main():
    app = QApplication(sys.argv)  # creates the PyQt5 application
    mn = Menu()  # Cretes the menu
    sys.exit(app.exec_())  # Close the application

if __name__ == '__main__':
    main()