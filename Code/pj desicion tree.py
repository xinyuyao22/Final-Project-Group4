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
#data.dropna(axis=1, inplace=True)
print (data.head())

X = data.drop(columns = (['target','first_active_month','card_id']))
y = data[['target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
#clf_gini = DecisionTreeClassifier(criterion="gini")
#clf_gini.fit(X_train, y_train)
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y)
#clf_gini = DecisionTreeClassifier(criterion="gini")
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X, training_scores_encoded)

y_pred_gini = clf_gini.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)

# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred_gini)
class_names = data.target.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()


#class_name = data[['target']].unique()


# display decision tree
dot_data = export_graphviz(clf_gini, filled=True, rounded=True,class_names=str(class_name), feature_names=data.drop(columns = (['target','first_active_month','card_id'])), out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_pdf("train_DT.pdf")
webbrowser.open_new(os.path.realpath('train_DT.pdf'))