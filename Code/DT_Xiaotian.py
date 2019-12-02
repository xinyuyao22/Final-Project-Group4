import pandas as pd
import numpy as np
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
from sklearn.metrics import mean_squared_error

data = pd.read_csv('train_fea_eng.csv')
print (data.head())

# if outliers not dropped
# bins = np.arange(-34, 20, 2)
# names = np.arange(-33, 19, 2)

bins = np.arange(-12.5, 12.5, 1)
names = np.arange(-12, 12, 1)
data['new_target'] = pd.cut(data['target'], bins, labels=names)

X = data.drop(columns = (['new_target','target','first_active_month','card_id']))
y = data['new_target']

#y=y.astype('int')
class_le = LabelEncoder()
y = class_le.fit_transform(y)

#print (data.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#%%-----------------------------------------------------------------------
# perform training with giniIndex.
# creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=0)

# performing training
clf_gini.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=0)

# Performing training
clf_entropy.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
# make predictions
# predicton on test using gini
y_pred_gini = clf_gini.predict(X_test)

# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(X_test)
#%%-----------------------------------------------------------------------
# calculate metrics gini model
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print('Mean_squared_error:', mean_squared_error(y_test, y_pred_gini))
print("\n")
print ('-'*80 + '\n')
# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print('Mean_squared_error:', mean_squared_error(y_test, y_pred_entropy))
print ('-'*80 + '\n')
#%%-----------------------------------------------------------------------
# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred_gini)
class_names = names
class_names = class_names.astype('str')
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 5}, yticklabels=df_cm.columns,
                 xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(25,25))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
plt.ylabel('True label',fontsize=100)
plt.xlabel('Predicted label',fontsize=100)
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------

# confusion matrix for entropy model
conf_matrix = confusion_matrix(y_test, y_pred_entropy)
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 5}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(25,25))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
plt.ylabel('True label',fontsize=100)
plt.xlabel('Predicted label',fontsize=100)
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
# display decision tree
dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=class_names, feature_names=X.columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_gini.pdf")
webbrowser.open_new(r'decision_tree_gini.pdf')

#%%-----------------------------------------------------------------------
# display decision tree

dot_data = export_graphviz(clf_entropy, filled=True, rounded=True, class_names=str(class_names), feature_names=X.columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_entropy.pdf")
webbrowser.open_new(r'decision_tree_entropy.pdf')