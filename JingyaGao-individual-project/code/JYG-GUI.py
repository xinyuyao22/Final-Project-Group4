#%%-----------------------------------------------------------------------
import sys

#from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel, QGridLayout, QCheckBox, QGroupBox
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt

from scipy import interp
from itertools import cycle


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

import warnings
# warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import random
import seaborn as sns
from sklearn.svm import SVC

import os
#%%-----------------------------------------------------------------------
font_size_window = 'font-size:15px'

class features(QMainWindow):
    def __init__(self):
        super(features, self).__init__()
        self.setWindowTitle("Feature Importance")
        self.initUi()

    def initUi(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        lay = QVBoxLayout(self.central_widget)

        label = QLabel(self)
        pixmap = QPixmap('1-features.png')
        label.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())

        lay.addWidget(label)
        self.show()

class KNN(QMainWindow):

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(KNN, self).__init__()
        self.Title = "K-Nearest Neighbors"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('KNN Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        # create a checkbox of each Features
        self.lblPercentTest = QLabel('number of K for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("25")

        self.btnExecute = QPushButton("Execute KNN")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.lblPercentTest, 0, 0)
        self.groupBox1Layout.addWidget(self.txtPercentTest, 0, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 1, 0)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #::--------------------------------------
        # Graphic 1 : Confusion Matrix
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : accuracy
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Accuracy with K')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBox2, 1, 0)
        self.layout.addWidget(self.groupBoxG2, 1, 1)


        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):
        vtest_per = int(self.txtPercentTest.text())

        self.ax1.clear()
        self.ax2.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        X1 = data1.drop(columns=(['new_target']))
        Y1 = data1['new_target']
        class_le = LabelEncoder()
        y1 = class_le.fit_transform(Y1)

        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=100, stratify=y1)
        stdsc = StandardScaler()
        stdsc.fit(X_train)
        X_train_std = stdsc.transform(X_train)
        X_test_std = stdsc.transform(X_test)
        self.clf = KNeighborsClassifier(n_neighbors=vtest_per)
        self.clf.fit(X_train_std, y_train)
        y_pred = self.clf.predict(X_test_std)

        conf_matrix = confusion_matrix(y_test, y_pred)

        self.ff_class_rep = classification_report(y_test, y_pred)
        self.txtResults.appendPlainText(self.ff_class_rep)

        self.ff_accuracy_score = accuracy_score(y_test, y_pred) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))

        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')
        class_names1 = data1['new_target'].unique()
        self.ax1.matshow(conf_matrix, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1, rotation=90)

        for i in range(len(class_names1)):
            for j in range(len(class_names1)):
                y_pred_score = self.clf.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::----------------------------------------------------------------
        # Graph2 -- Accuracy
        #::-----------------------------------------------------------------

        k_range = range(1, 50, 3)
        k_scores = []
        for k in k_range:
            knn = KNeighborsClassifier(k)

            scores = cross_val_score(knn, X1, y1, cv=10, scoring='accuracy')
            k_scores.append(scores.mean())

        self.ax2.plot(k_range, k_scores)
        self.ax2.scatter(k_range, k_scores)
        self.ax2.set_title("KNN vs Accuracy")
        self.ax2.set_xlabel('Value of K for KNN')
        self.ax2.set_ylabel('Cross_Validation Accuracy')

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

class SVM(QMainWindow):

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(SVM, self).__init__()
        super().__init__()
        self.Title = "Support Vector Machine"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('SVM Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.lbl_1 = QLabel('number of first feature(0-6):')
        self.lbl_1.adjustSize()
        self.txt_1 = QLineEdit(self)
        self.txt_1.setText("0")

        self.lbl_2 = QLabel('number of second feature(0-6):')
        self.lbl_2.adjustSize()
        self.txt_2 = QLineEdit(self)
        self.txt_2.setText("4")

        self.btnExecute = QPushButton("Create Plot")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.lbl_1, 0, 0)
        self.groupBox1Layout.addWidget(self.txt_1, 0, 1)
        self.groupBox1Layout.addWidget(self.lbl_2, 1, 0)
        self.groupBox1Layout.addWidget(self.txt_2, 1, 1)
        self.groupBox1Layout.addWidget(self.btnExecute, 2, 0)


        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        #::--------------------------------------
        # Graphic 1 : original scatter
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Scatter before SVM')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : accuracy
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Accuracy with Samples')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------------
        # scatter after svm
        #::-------------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('Scatter after SVM')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)

        self.groupBoxG3Layout.addWidget(self.canvas3)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBoxG1, 0, 1)
        self.layout.addWidget(self.groupBoxG2, 1, 0)
        self.layout.addWidget(self.groupBoxG3, 1, 1)


        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):

        f1 = int(self.txt_1.text())
        f2 = int(self.txt_2.text())


        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()


        X = data2.iloc[:, [f1, f2]]
        y = data2['new_target']
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        data = pd.merge(X, y, left_index=True, right_index=True, how='outer')
        data.columns = ['x1', 'x2', 'y']
        h = 1
        x_min, x_max = data.x1.min() - 1, data.x1.max() + 1
        y_min, y_max = data.x2.min() - 1, data.x2.max() + 1


        self.ax1.scatter(x=data.x1, y=data.x2, c=data.y)
        self.ax1.set_xlabel("Label for X")
        self.ax1.set_ylabel("Label for y")
        self.ax1.grid(True)
        plt.show()

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


        X = data[['x1', 'x2']]
        y = data.y
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=13)
        self.clf = SVC(C=6, kernel='rbf')
        self.clf.fit(X_train, y_train)
        y_pre = self.clf.predict(X)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        color = ["#E74C3C", "#8E44AD", "#3498DB", "#27AE60", "#F1C40F", "#E67E22", "#BDC3C7"]
        self.ax3.scatter(data.x1, y=data.x2, c=y_pre)
        self.ax3.contour(xx, yy, Z, colors=color, alpha=0.2)
        self.ax3.set_xlabel("Label for X")
        self.ax3.set_ylabel("Label for y")
        self.ax3.grid(True)
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()

        scores = []
        for m in range(3, X_train.size):
            self.clf.fit(X_train[:m], y_train[:m])
            y_train_predict = self.clf.predict(X_train[:m])
            y_val_predict = self.clf.predict(X_val)
            scores.append(accuracy_score(y_train_predict, y_train[:m]))
        self.ax2.plot(range(3, X_train.size), scores, c='green', alpha=0.6)
        self.ax2.set_xlabel("Samples")
        self.ax2.set_ylabel("Accuracy")
        self.ax2.grid(True)
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

class Kmeans(QMainWindow):

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(Kmeans, self).__init__()
        super().__init__()
        self.Title = "K-Means Clustering"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('K-Means Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.lbl_1 = QLabel('number of first feature(0-6):')
        self.lbl_1.adjustSize()
        self.txt_1 = QLineEdit(self)
        self.txt_1.setText("0")

        self.lbl_2 = QLabel('number of second feature(0-6):')
        self.lbl_2.adjustSize()
        self.txt_2 = QLineEdit(self)
        self.txt_2.setText("4")

        self.btnExecute = QPushButton("Create Plot")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.lbl_1, 0, 0)
        self.groupBox1Layout.addWidget(self.txt_1, 0, 1)
        self.groupBox1Layout.addWidget(self.lbl_2, 0, 2)
        self.groupBox1Layout.addWidget(self.txt_2, 0, 3)
        self.groupBox1Layout.addWidget(self.btnExecute, 1, 0)


        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        #::--------------------------------------
        # Graphic 1 : original scatter
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Scatter before K-Means')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : after
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Scatter after K-Means')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1, 0, 1)
        self.layout.addWidget(self.groupBoxG1, 1, 0)
        self.layout.addWidget(self.groupBoxG2, 1, 1)


        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):

        f1 = int(self.txt_1.text())
        f2 = int(self.txt_2.text())


        self.ax1.clear()
        self.ax2.clear()


        X = data2.iloc[:, [f1, f2]]
        y = data2['new_target']
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        data = pd.merge(X, y, left_index=True, right_index=True, how='outer')
        data.columns = ['x1', 'x2', 'y']
        h = 1
        x_min, x_max = data.x1.min() - 1, data.x1.max() + 1
        y_min, y_max = data.x2.min() - 1, data.x2.max() + 1


        self.ax1.scatter(x=data.x1, y=data.x2, c=data.y)
        self.ax1.set_xlabel("X1")
        self.ax1.set_ylabel("X2")
        self.ax1.grid(True)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        self.estimator = KMeans(n_clusters=3)
        self.estimator.fit(X)
        label_pred = self.estimator.labels_

        x0 = X[label_pred == 0]
        x1 = X[label_pred == 1]
        x2 = X[label_pred == 2]
        self.ax2.scatter(x0.iloc[:, 0], x0.iloc[:, 1], c="red", marker='o', label='label0')
        self.ax2.scatter(x1.iloc[:, 0], x1.iloc[:, 1], c="green", marker='*', label='label1')
        self.ax2.scatter(x2.iloc[:, 0], x2.iloc[:, 1], c="blue", marker='+', label='label2')
        self.ax2.set_xlabel('X1')
        self.ax2.set_ylabel('X2')
        self.ax2.grid(True)
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

class AGNES(QMainWindow):

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(AGNES, self).__init__()
        super().__init__()
        self.Title = "Agglomerative Nesting (Hierarchical Clustering)"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('AGNES Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.lbl_1 = QLabel('number of first feature(0-6):')
        self.lbl_1.adjustSize()
        self.txt_1 = QLineEdit(self)
        self.txt_1.setText("0")

        self.lbl_2 = QLabel('number of second feature(0-6):')
        self.lbl_2.adjustSize()
        self.txt_2 = QLineEdit(self)
        self.txt_2.setText("4")

        self.btnExecute = QPushButton("Create Plot")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.lbl_1, 0, 0)
        self.groupBox1Layout.addWidget(self.txt_1, 0, 1)
        self.groupBox1Layout.addWidget(self.lbl_2, 0, 2)
        self.groupBox1Layout.addWidget(self.txt_2, 0, 3)
        self.groupBox1Layout.addWidget(self.btnExecute, 1, 0)


        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        #::--------------------------------------
        # Graphic 1 : original scatter
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Scatter before AGNES')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : after
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Scatter after AGNES')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1, 0, 1)
        self.layout.addWidget(self.groupBoxG1, 1, 0)
        self.layout.addWidget(self.groupBoxG2, 1, 1)


        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):

        f1 = int(self.txt_1.text())
        f2 = int(self.txt_2.text())


        self.ax1.clear()
        self.ax2.clear()


        X = data2.iloc[:, [f1, f2]]
        y = data2['new_target']
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        data = pd.merge(X, y, left_index=True, right_index=True, how='outer')
        data.columns = ['x1', 'x2', 'y']
        h = 1
        x_min, x_max = data.x1.min() - 1, data.x1.max() + 1
        y_min, y_max = data.x2.min() - 1, data.x2.max() + 1


        self.ax1.scatter(x=data.x1, y=data.x2, c=data.y)
        self.ax1.set_xlabel("X1")
        self.ax1.set_ylabel("X2")
        self.ax1.grid(True)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        self.clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)
        self.clustering.fit(X)
        label_pred = self.clustering.labels_

        x0 = X[label_pred == 0]
        x1 = X[label_pred == 1]
        x2 = X[label_pred == 2]
        self.ax2.scatter(x0.iloc[:, 0], x0.iloc[:, 1], c="red", marker='o', label='label0')
        self.ax2.scatter(x1.iloc[:, 0], x1.iloc[:, 1], c="green", marker='*', label='label1')
        self.ax2.scatter(x2.iloc[:, 0], x2.iloc[:, 1], c="blue", marker='+', label='label2')
        self.ax2.set_xlabel('X1')
        self.ax2.set_ylabel('X2')
        self.ax2.grid(True)
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

class dbscan(QMainWindow):

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(dbscan, self).__init__()
        super().__init__()
        self.Title = "Density-Based Spatial Clustering of Applications with Noise"
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('DBSCAN Features')
        self.groupBox1Layout= QGridLayout()   # Grid
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.lbl_1 = QLabel('number of first feature(0-6):')
        self.lbl_1.adjustSize()
        self.txt_1 = QLineEdit(self)
        self.txt_1.setText("0")

        self.lbl_2 = QLabel('number of second feature(0-6):')
        self.lbl_2.adjustSize()
        self.txt_2 = QLineEdit(self)
        self.txt_2.setText("4")

        self.btnExecute = QPushButton("Create Plot")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.lbl_1, 0, 0)
        self.groupBox1Layout.addWidget(self.txt_1, 0, 1)
        self.groupBox1Layout.addWidget(self.lbl_2, 0, 2)
        self.groupBox1Layout.addWidget(self.txt_2, 0, 3)
        self.groupBox1Layout.addWidget(self.btnExecute, 1, 0)


        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        #::--------------------------------------
        # Graphic 1 : original scatter
        #::--------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Scatter before DBSCAN')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::---------------------------------------
        # Graphic 2 : after
        #::---------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Scatter after DBSCAN')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1, 0, 1)
        self.layout.addWidget(self.groupBoxG1, 1, 0)
        self.layout.addWidget(self.groupBoxG2, 1, 1)


        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):

        f1 = int(self.txt_1.text())
        f2 = int(self.txt_2.text())


        self.ax1.clear()
        self.ax2.clear()


        X = data2.iloc[:, [f1, f2]]
        y = data2['new_target']
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        data = pd.merge(X, y, left_index=True, right_index=True, how='outer')
        data.columns = ['x1', 'x2', 'y']
        h = 1
        x_min, x_max = data.x1.min() - 1, data.x1.max() + 1
        y_min, y_max = data.x2.min() - 1, data.x2.max() + 1


        self.ax1.scatter(x=data.x1, y=data.x2, c=data.y)
        self.ax1.set_xlabel("X1")
        self.ax1.set_ylabel("X2")
        self.ax1.grid(True)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        self.dbscan_clf = DBSCAN(eps=0.4, min_samples=9)
        self.dbscan_clf.fit(X)
        label_pred = self.dbscan_clf.labels_

        x0 = X[label_pred == 0]
        x1 = X[label_pred == -1]
        x2 = X[label_pred == 1]
        self.ax2.scatter(x0.iloc[:, 0], x0.iloc[:, 1], c="red", marker='o', label='label0')
        self.ax2.scatter(x1.iloc[:, 0], x1.iloc[:, 1], c="green", marker='*', label='label1')
        self.ax2.scatter(x2.iloc[:, 0], x2.iloc[:, 1], c="blue", marker='+', label='label2')
        self.ax2.set_xlabel('X1')
        self.ax2.set_ylabel('X2')
        self.ax2.grid(True)
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'JYG-6103-GUI'
        self.width = 500
        self.height = 300
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightblue')

        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('Feature Importance')
        MLModelMenu = mainMenu.addMenu('ML Models')

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), ' &Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)
        #::----------------------------------------
        # Feature Importance
        #::----------------------------------------
        EDA1Button = QAction("features", self)
        EDA1Button.setStatusTip('Presents the initial datasets')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)
        #::--------------------------------------------------
        # ML Models for prediction
        # 5 models
        #   KNN; SVM; KMEANS; AGNES; DBSCAN
        #::--------------------------------------------------
        # KNN
        #::--------------------------------------------------
        MLModel1Button = QAction(QIcon(), 'KNN', self)
        MLModel1Button.setStatusTip('KNN ')
        MLModel1Button.triggered.connect(self.ML1)

        #::------------------------------------------------------
        # SVM
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon(), 'SVM', self)
        MLModel2Button.setStatusTip('SVM ')
        MLModel2Button.triggered.connect(self.ML2)

        #::--------------------------------------------------
        # Kmeans
        #::--------------------------------------------------
        MLModel3Button = QAction(QIcon(), 'Kmeans', self)
        MLModel3Button.setStatusTip('Kmeans ')
        MLModel3Button.triggered.connect(self.ML3)

        #::------------------------------------------------------
        # AGNES
        #::------------------------------------------------------
        MLModel4Button = QAction(QIcon(), 'AGNES', self)
        MLModel4Button.setStatusTip('AGNES ')
        MLModel4Button.triggered.connect(self.ML4)

        #::--------------------------------------------------
        # DBSCAN
        #::--------------------------------------------------
        MLModel5Button = QAction(QIcon(), 'DBSCAN', self)
        MLModel5Button.setStatusTip('DBSCAN ')
        MLModel5Button.triggered.connect(self.ML5)

        #::------------------------------------------------------

        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel2Button)
        MLModelMenu.addAction(MLModel3Button)
        MLModelMenu.addAction(MLModel4Button)
        MLModelMenu.addAction(MLModel5Button)

        self.dialogs = list()
        self.show()

    def EDA1(self):
        dialog = features()
        self.dialogs.append(dialog)
        dialog.show()

    def ML1(self):
        dialog = KNN()
        self.dialogs.append(dialog)
        dialog.show()

    def ML2(self):
        dialog = SVM()
        self.dialogs.append(dialog)
        dialog.show()

    def ML3(self):
        dialog = Kmeans()
        self.dialogs.append(dialog)
        dialog.show()

    def ML4(self):
        dialog = AGNES()
        self.dialogs.append(dialog)
        dialog.show()

    def ML5(self):
        dialog = dbscan()
        self.dialogs.append(dialog)
        dialog.show()

def main():
    app = QApplication(sys.argv)  # creates the PyQt5 application
    mn = App()  # Cretes the menu
    sys.exit(app.exec_())  # Close the application

def data_set():
    global data1
    global data2
    global features_list

    warnings.filterwarnings("ignore")
    data0 = pd.read_csv('newData_select.csv')
    data1 = data0.sample(frac=0.07, random_state=23)
    data2 = data0.sample(frac=0.01, random_state=1900)
    features_list = ["auth_purchase_month_std", "new_merchant_id_nunique",
                     "purchase_amount_count_mean", "month_lag_std",
                     "purchase_amount_count_std", "auth_merchant_id_nunique",
                     "new_purchase_month_mean"]

if __name__ == '__main__':
    data_set()
    main()
