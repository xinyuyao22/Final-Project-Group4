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
import matplotlib
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
def heatmap(data, row_labels, col_labels, ax=None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

class GraphWParamsClass(QMainWindow):
    send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(GraphWParamsClass, self).__init__()

        self.Title = 'Title : Histogram Target Count '
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
        self.v = "24 Categories"
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout

        self.groupBox1 = QGroupBox('Number of Categories')
        self.groupBox1Layout = QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.groupBox2 = QGroupBox('Accuracy and MSE')
        self.groupBox2Layout = QGridLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.label1 = QLabel("Accuracy: 33.93")
        self.label2 = QLabel("MSE: 2.89")
        self.label3 = QLabel("Accuracy: 95.89")
        self.label4 = QLabel("MSE: 2.63")

        self.groupBox2Layout.addWidget(self.label1, 0, 0)
        self.groupBox2Layout.addWidget(self.label2, 1, 0)
        self.groupBox2Layout.addWidget(self.label3, 0, 1)
        self.groupBox2Layout.addWidget(self.label4, 1, 1)

        self.groupBox3 = QGroupBox('Graphic')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        # Radio buttons are create to be added to the second group

        self.b1 = QRadioButton("24 Categories")
        self.b1.setChecked(True)
        self.b1.toggled.connect(self.onClicked)

        self.b2 = QRadioButton("3 Categories")
        self.b2.toggled.connect(self.onClicked)

        self.buttonlabel = QLabel(self.v+' is selected')

        self.groupBox1Layout.addWidget(self.b1)
        self.groupBox1Layout.addWidget(self.b2)
        self.groupBox1Layout.addWidget(self.buttonlabel)

        # figure and canvas figure to draw the graph is created to
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        # Canvas is added to the third group box
        self.groupBox3Layout.addWidget(self.canvas)

        # Adding to the main layout the groupboxes
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(600, 500)                         # Resize the window
        self.onClicked()


    def onClicked(self):

        # Figure is cleared to create the new graph with the choosen parameters
        self.ax1.clear()

        # the buttons are inspect to indicate which one is checked.
        # vcolor is assigned the chosen color
        if self.b1.isChecked():
            self.v = self.b1.text()
            df_cm = pd.read_csv('RF_gini_cm.csv')
            im = self.ax1.imshow(df_cm)

            # We want to show all ticks...
            self.ax1.set_xticks(np.arange(1, df_cm.shape[0] + 1))
            self.ax1.set_yticks(np.arange(df_cm.shape[0]))
            # ... and label them with the respective list entries
            self.ax1.set_xticklabels(np.arange(-12, 12))
            self.ax1.set_yticklabels(np.arange(-12, 12))

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


        if self.b2.isChecked():
            self.v = self.b2.text()
            df_cm = pd.read_csv('RF_gini_cm_improve2.csv')
            im= heatmap(df_cm.iloc[:, 1:df_cm.shape[0] + 1], ['[-12, -4)', '[-4, 4)', '[4, 12)'], ['[-12, -4)', '[-4, 4)', '[4, 12)'], ax=self.ax1)
            texts = annotate_heatmap(im, valfmt="{x:.1f}")

        # the label that displays the selected option
        self.buttonlabel.setText(self.v+' is selected')

        # show the plot
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

class NB(QMainWindow):
    send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(NB, self).__init__()

        self.Title = 'Title : Naive Bayes '
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QVBoxLayout (Vertical Layout )
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.v = "24 Categories"
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout

        self.groupBox1 = QGroupBox('Number of Categories')
        self.groupBox1Layout = QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.groupBox2 = QGroupBox('Accuracy and MSE')
        self.groupBox2Layout = QGridLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.label1 = QLabel("Accuracy: 33.46")
        self.label2 = QLabel("MSE: 3.06")
        self.label3 = QLabel("Accuracy: 87.63")

        self.groupBox2Layout.addWidget(self.label1, 0, 0)
        self.groupBox2Layout.addWidget(self.label2, 1, 0)
        self.groupBox2Layout.addWidget(self.label3, 0, 1)

        self.groupBox3 = QGroupBox('Graphic')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        # Radio buttons are create to be added to the second group

        self.b1 = QRadioButton("24 Categories")
        self.b1.setChecked(True)
        self.b1.toggled.connect(self.onClicked)

        self.b2 = QRadioButton("3 Categories")
        self.b2.toggled.connect(self.onClicked)

        self.buttonlabel = QLabel(self.v+' is selected')

        self.groupBox1Layout.addWidget(self.b1)
        self.groupBox1Layout.addWidget(self.b2)
        self.groupBox1Layout.addWidget(self.buttonlabel)

        # figure and canvas figure to draw the graph is created to
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        # Canvas is added to the third group box
        self.groupBox3Layout.addWidget(self.canvas)

        # Adding to the main layout the groupboxes
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(600, 500)                         # Resize the window
        self.onClicked()


    def onClicked(self):

        # Figure is cleared to create the new graph with the choosen parameters
        self.ax1.clear()

        # the buttons are inspect to indicate which one is checked.
        # vcolor is assigned the chosen color
        if self.b1.isChecked():
            self.v = self.b1.text()
            df_cm = pd.read_csv('NB_cm.csv')
            im = self.ax1.imshow(df_cm)

            # We want to show all ticks...
            self.ax1.set_xticks(np.arange(1, df_cm.shape[0] + 1))
            self.ax1.set_yticks(np.arange(df_cm.shape[0]))
            # ... and label them with the respective list entries
            self.ax1.set_xticklabels(np.arange(-12, 12))
            self.ax1.set_yticklabels(np.arange(-12, 12))

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

        if self.b2.isChecked():
            self.v = self.b2.text()
            df_cm = pd.read_csv('NB_cm_improve2.csv')
            im = heatmap(df_cm.iloc[:, 1:df_cm.shape[0] + 1], ['[-12, -4)', '[-4, 4)', '[4, 12)'],
                         ['[-12, -4)', '[-4, 4)', '[4, 12)'], ax=self.ax1)
            texts = annotate_heatmap(im, valfmt="{x:.1f}")

        # the label that displays the selected option
        self.buttonlabel.setText(self.v + ' is selected')

        # show the plot
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

class KNN(QMainWindow):
    send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(KNN, self).__init__()

        self.Title = 'Title : K-Nearest Neighbor '
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QVBoxLayout (Vertical Layout )
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.v = "24 Categories"
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout

        self.groupBox1 = QGroupBox('Number of Categories')
        self.groupBox1Layout = QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.groupBox2 = QGroupBox('Accuracy and MSE')
        self.groupBox2Layout = QGridLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.label1 = QLabel("Accuracy: 22.18")
        self.label2 = QLabel("MSE: 5.84")
        self.label3 = QLabel("Accuracy: 95.32")
        #self.label4 = QLabel("MSE: 0.05")

        self.groupBox2Layout.addWidget(self.label1, 0, 0)
        self.groupBox2Layout.addWidget(self.label2, 1, 0)
        self.groupBox2Layout.addWidget(self.label3, 0, 1)
        #self.groupBox2Layout.addWidget(self.label4, 1, 1)

        self.groupBox3 = QGroupBox('Graphic')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        # Radio buttons are create to be added to the second group

        self.b1 = QRadioButton("24 Categories")
        self.b1.setChecked(True)
        self.b1.toggled.connect(self.onClicked)

        self.b2 = QRadioButton("3 Categories")
        self.b2.toggled.connect(self.onClicked)

        self.buttonlabel = QLabel(self.v+' is selected')

        self.groupBox1Layout.addWidget(self.b1)
        self.groupBox1Layout.addWidget(self.b2)
        self.groupBox1Layout.addWidget(self.buttonlabel)

        # figure and canvas figure to draw the graph is created to
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        # Canvas is added to the third group box
        self.groupBox3Layout.addWidget(self.canvas)

        # Adding to the main layout the groupboxes
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(600, 500)                         # Resize the window
        self.onClicked()


    def onClicked(self):

        # Figure is cleared to create the new graph with the choosen parameters
        self.ax1.clear()

        # the buttons are inspect to indicate which one is checked.
        # vcolor is assigned the chosen color
        if self.b1.isChecked():
            self.v = self.b1.text()
            df_cm = pd.read_csv('KNN_cm.csv')
            im = self.ax1.imshow(df_cm)

            # We want to show all ticks...
            self.ax1.set_xticks(np.arange(1, df_cm.shape[0] + 1))
            self.ax1.set_yticks(np.arange(df_cm.shape[0]))
            # ... and label them with the respective list entries
            self.ax1.set_xticklabels(np.arange(-12, 12))
            self.ax1.set_yticklabels(np.arange(-12, 12))

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

        if self.b2.isChecked():
            self.v = self.b2.text()
            df_cm = pd.read_csv('KNN_cm_improve2.csv')
            im = heatmap(df_cm.iloc[:, 1:df_cm.shape[0] + 1], ['[-12, -4)', '[-4, 4)', '[4, 12)'],
                         ['[-12, -4)', '[-4, 4)', '[4, 12)'], ax=self.ax1)
            texts = annotate_heatmap(im, valfmt="{x:.1f}")

        # the label that displays the selected option
        self.buttonlabel.setText(self.v + ' is selected')

        # show the plot
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

class SVM(QMainWindow):
    send_fig = pyqtSignal(str)  # To manage the signals PyQT manages the communication

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #::--------------------------------------------------------
        super(SVM, self).__init__()

        self.Title = 'Title : Support Vector Machine '
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  We create the type of layout QVBoxLayout (Vertical Layout )
        #  This type of layout comes from QWidget
        #::--------------------------------------------------------------
        self.v = "24 Categories"
        self.setWindowTitle(self.Title)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)   # Creates vertical layout

        self.groupBox1 = QGroupBox('Number of Categories')
        self.groupBox1Layout = QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.groupBox2 = QGroupBox('Accuracy and MSE')
        self.groupBox2Layout = QGridLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.label1 = QLabel("Accuracy: ")
        self.label2 = QLabel("MSE: ")
        self.label3 = QLabel("Accuracy: ")
        #self.label4 = QLabel("MSE: 0.05")

        self.groupBox2Layout.addWidget(self.label1, 0, 0)
        self.groupBox2Layout.addWidget(self.label2, 1, 0)
        self.groupBox2Layout.addWidget(self.label3, 0, 1)
        #self.groupBox2Layout.addWidget(self.label4, 1, 1)

        self.groupBox3 = QGroupBox('Graphic')
        self.groupBox3Layout = QVBoxLayout()
        self.groupBox3.setLayout(self.groupBox3Layout)

        # Radio buttons are create to be added to the second group

        self.b1 = QRadioButton("24 Categories")
        self.b1.setChecked(True)
        self.b1.toggled.connect(self.onClicked)

        self.b2 = QRadioButton("3 Categories")
        self.b2.toggled.connect(self.onClicked)

        self.buttonlabel = QLabel(self.v+' is selected')

        self.groupBox1Layout.addWidget(self.b1)
        self.groupBox1Layout.addWidget(self.b2)
        self.groupBox1Layout.addWidget(self.buttonlabel)

        # figure and canvas figure to draw the graph is created to
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        # Canvas is added to the third group box
        self.groupBox3Layout.addWidget(self.canvas)

        # Adding to the main layout the groupboxes
        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)
        self.layout.addWidget(self.groupBox3)

        self.setCentralWidget(self.main_widget)       # Creates the window with all the elements
        self.resize(600, 500)                         # Resize the window
        self.onClicked()


    def onClicked(self):

        # Figure is cleared to create the new graph with the choosen parameters
        self.ax1.clear()

        # the buttons are inspect to indicate which one is checked.
        # vcolor is assigned the chosen color
        if self.b1.isChecked():
            self.v = self.b1.text()
            df_cm = pd.read_csv('SVM_cm.csv')
            im = self.ax1.imshow(df_cm)

            # We want to show all ticks...
            self.ax1.set_xticks(np.arange(1, df_cm.shape[0] + 1))
            self.ax1.set_yticks(np.arange(df_cm.shape[0]))
            # ... and label them with the respective list entries
            self.ax1.set_xticklabels(np.arange(-12, 12))
            self.ax1.set_yticklabels(np.arange(-12, 12))

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

        if self.b2.isChecked():
            self.v = self.b2.text()
            df_cm = pd.read_csv('SVM_cm_improve2.csv')
            im = heatmap(df_cm.iloc[:, 1:df_cm.shape[0] + 1], ['[-12, -4)', '[-4, 4)', '[4, 12)'],
                         ['[-12, -4)', '[-4, 4)', '[4, 12)'], ax=self.ax1)
            texts = annotate_heatmap(im, valfmt="{x:.1f}")

        # the label that displays the selected option
        self.buttonlabel.setText(self.v + ' is selected')

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

        exampleGraphic5 = QAction("Support Vector Machine", self)
        exampleGraphic5.setStatusTip('Example of Graphic')
        exampleGraphic5.triggered.connect(self.SVM)

        exampleWin.addAction(exampleGraphic5)

        exampleGraphic4 = QAction("K-Nearest Neighbor", self)
        exampleGraphic4.setStatusTip('Example of Graphic')
        exampleGraphic4.triggered.connect(self.KNN)

        exampleWin.addAction(exampleGraphic4)

        exampleGraphic3 = QAction("Naive Bayes", self)
        exampleGraphic3.setStatusTip('Example of Graphic')
        exampleGraphic3.triggered.connect(self.NB)

        exampleWin.addAction(exampleGraphic3)

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

    def NB(self):
        dialog = NB()
        self.dialogs.append(dialog) # Apppends to the list of dialogs
        dialog.show()

    def KNN(self):
        dialog = KNN()
        self.dialogs.append(dialog) # Apppends to the list of dialogs
        dialog.show()

    def SVM(self):
        dialog = SVM()
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