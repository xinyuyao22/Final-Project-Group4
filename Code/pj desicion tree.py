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
from sklearn.metrics import mean_squared_error

data = pd.read_csv('train_fea_eng.csv',sep = ',')
#data.dropna(axis=1, inplace=True)
print (data.head())

bins = [-34,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,
       -14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
names = [-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,
       -14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
data['new_target'] = pd.cut(data['target'], bins, labels=names)

X = data.drop(columns = (['target','first_active_month','card_id']))
y = data['new_target']

y=y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
clf_gini = DecisionTreeClassifier(criterion="gini")
clf_gini.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print (mean_squared_error(y_test, y_pred_gini))

class_name = data.iloc[:,-1].unique()

# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred_gini)
df_cm = pd.DataFrame(conf_matrix, index=class_name, columns=class_name)

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

# display decision tree
class_name = data.iloc[:,-1].unique()
dot_data = export_graphviz(clf_gini, filled=True, rounded=True,class_names=str(class_name), feature_names=data.drop(columns = (['target','first_active_month','card_id'])).columns, out_file=None)

#dot_data = export_graphviz(clf_gini, filled=True, rounded=True,class_names=str(class_name), feature_names=data.drop(columns = (['target','first_active_month','card_id'])), out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_pdf("train_DT.pdf")
webbrowser.open_new(os.path.realpath('train_DT.pdf'))