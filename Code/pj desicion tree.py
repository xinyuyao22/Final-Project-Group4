import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression




data = pd.read_csv('train_fea_eng.csv',sep = ',')
data.dropna(axis=1, inplace=True)
print (data.head())

X = data.drop(columns = (['target','first_active_month','card_id']))
y = data[['target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
clf_gini = DecisionTreeClassifier(criterion="gini")
#clf_gini.fit(X_train, y_train)
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y)
clf = LogisticRegression()
clf_gini.fit(X, training_scores_encoded)
class_name = data[['target']].unique()
dot_data = export_graphviz(clf_gini, filled=True, rounded=True,class_names=str(class_name), feature_names=data.drop(columns = (['target','first_active_month','card_id'])), out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("train_DT.pdf")
webbrowser.open_new(os.path.realpath('train_DT.pdf'))