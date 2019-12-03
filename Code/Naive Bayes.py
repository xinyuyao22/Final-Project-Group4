import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

data = pd.read_csv('feature_select.csv',sep = ',')
X = data.drop(columns = (['new_target']))
Y = data[['new_target']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

clf = GaussianNB()
clf.fit(X_train, y_train)

# predicton on test
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

print ('Mean_squared_error:', mean_squared_error(y_test, y_pred))

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)

# confusion matrix


conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['new_target'].unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)


plt.figure(figsize=(25,25))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
plt.ylabel('True label',fontsize=100)
plt.xlabel('Predicted label',fontsize=100)
plt.tight_layout()
df_cm.to_csv('NB_cm.csv')
plt.show()

#Ranging by interval 4
data = pd.read_csv('train_fea_eng.csv')
bins = np.arange(-12.5, 12.5, 4)
names = np.arange(-12, 12, 4)
data['new_target'] = pd.cut(data['target'], bins, labels=names)

X = data.drop(columns = (['new_target','target','first_active_month','card_id']))
Y = data['new_target']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

clf = GaussianNB()
clf.fit(X_train, y_train)

# predicton on test
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

print ('Mean_squared_error:', mean_squared_error(y_test, y_pred))

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
# confusion matrix


conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['new_target'].unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)


plt.figure(figsize=(25,25))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
plt.ylabel('True label',fontsize=100)
plt.xlabel('Predicted label',fontsize=100)
plt.tight_layout()
df_cm.to_csv('NB_cm_improve.csv')
plt.show()


#Ranging by interval 8
data = pd.read_csv('train_fea_eng.csv')

bins = np.arange(-12.5, 12.5, 8)
names = np.arange(-12, 12, 8)
data['new_target'] = pd.cut(data['target'], bins, labels=names)

X = data.drop(columns = (['new_target','target','first_active_month','card_id']))
Y = data['new_target']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

clf = GaussianNB()
clf.fit(X_train, y_train)

# predicton on test
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

print ('Mean_squared_error:', mean_squared_error(y_test, y_pred))

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
# confusion matrix


conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['new_target'].unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)


plt.figure(figsize=(25,25))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
plt.ylabel('True label',fontsize=100)
plt.xlabel('Predicted label',fontsize=100)
plt.tight_layout()
df_cm.to_csv('NB_cm_improve2.csv')
plt.show()