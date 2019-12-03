#%%-----------------------------------------------------------------------
# Importing the required packages

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error


data = pd.read_csv('feature_select.csv')


X = data.drop(columns = (['new_target']))
Y = data['new_target']
#%%-----------------------------------------------------------------------
# data preprocessing
# encode the target variable
class_le = LabelEncoder()

y = class_le.fit_transform(Y)

#%%-----------------------------------------------------------------------
# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

#%%-----------------------------------------------------------------------
# data preprocessing
# standardize the data
stdsc = StandardScaler()

stdsc.fit(X_train)

X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

#%%-----------------------------------------------------------------------
# perform training
# creating the classifier object
clf = KNeighborsClassifier(n_neighbors=3)

# performing training
clf.fit(X_train_std, y_train)

#%%-----------------------------------------------------------------------
# make predictions

# predicton on test
y_pred = clf.predict(X_test_std)

#%%-----------------------------------------------------------------------
# calculate metrics

print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print (mean_squared_error(y_test, y_pred))

#%%-----------------------------------------------------------------------
# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['new_target'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )


plt.figure(figsize=(25,25))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
plt.ylabel('True label',fontsize=100)
plt.xlabel('Predicted label',fontsize=100)
plt.tight_layout()
df_cm.to_csv('KNN_cm.csv')
plt.show()

#Ranging by interval 4
data = pd.read_csv('feature_select_improve.csv')
X = data.drop(columns = (['new_target']))
Y = data['new_target']
#%%-----------------------------------------------------------------------
# data preprocessing
# encode the target variable
class_le = LabelEncoder()

y = class_le.fit_transform(Y)

#%%-----------------------------------------------------------------------
# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

#%%-----------------------------------------------------------------------
# data preprocessing
# standardize the data
stdsc = StandardScaler()

stdsc.fit(X_train)

X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

#%%-----------------------------------------------------------------------
# perform training
# creating the classifier object
clf = KNeighborsClassifier(n_neighbors=3)

# performing training
clf.fit(X_train_std, y_train)

#%%-----------------------------------------------------------------------
# make predictions

# predicton on test
y_pred = clf.predict(X_test_std)

#%%-----------------------------------------------------------------------
# calculate metrics

print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print ('Mean_squared_error:', mean_squared_error(y_test, y_pred))

#%%-----------------------------------------------------------------------
# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['new_target'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )


plt.figure(figsize=(25,25))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
plt.ylabel('True label',fontsize=100)
plt.xlabel('Predicted label',fontsize=100)
plt.tight_layout()
df_cm.to_csv('KNN_cm_improve.csv')
plt.show()




#Ranging by interval 8
data = pd.read_csv('feature_select_improve2.csv')


X = data.drop(columns = (['new_target']))
Y = data['new_target']
#%%-----------------------------------------------------------------------
# data preprocessing
# encode the target variable
class_le = LabelEncoder()

y = class_le.fit_transform(Y)

#%%-----------------------------------------------------------------------
# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

#%%-----------------------------------------------------------------------
# data preprocessing
# standardize the data
stdsc = StandardScaler()

stdsc.fit(X_train)

X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

#%%-----------------------------------------------------------------------
# perform training
# creating the classifier object
clf = KNeighborsClassifier(n_neighbors=3)

# performing training
clf.fit(X_train_std, y_train)

#%%-----------------------------------------------------------------------
# make predictions

# predicton on test
y_pred = clf.predict(X_test_std)

#%%-----------------------------------------------------------------------
# calculate metrics

print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print ('Mean_squared_error:', mean_squared_error(y_test, y_pred))

#%%-----------------------------------------------------------------------
# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['new_target'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )


plt.figure(figsize=(25,25))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
plt.ylabel('True label',fontsize=100)
plt.xlabel('Predicted label',fontsize=100)
plt.tight_layout()
df_cm.to_csv('KNN_cm_improve2.csv')
plt.show()