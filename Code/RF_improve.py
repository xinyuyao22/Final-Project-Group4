#%%-----------------------------------------------------------------------
# Importing the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error


#%%-----------------------------------------------------------------------
# import Dataset
# read data as panda dataframe
data = pd.read_csv('train_fea_eng.csv')
bins = np.arange(-12.5, 12.5, 4)
names = np.arange(-12, 12, 4)
data['new_target'] = pd.cut(data['target'], bins, labels=names)

#%%-----------------------------------------------------------------------
#clean the dataset


# drop unnnecessary columns
#data.drop(['target','first_active_month','card_id'], axis=1, inplace=True)

# encode target variable
#data['new_target'] = data['new_target'].map({'M': 1, 'B': 0})
#%%-----------------------------------------------------------------------
#split the dataset
# separate the predictor and target variable
X = data.drop(columns = (['new_target','target','first_active_month','card_id']))
Y = data['new_target']
# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
#%%-----------------------------------------------------------------------
#perform training with random forest with all columns
# specify random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# perform training
clf.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
#plot feature importances
# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, data.drop(columns = (['new_target','target','first_active_month','card_id'])).columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
#f_importances.to_csv('RF_feature.csv')
plt.show()

#%%-----------------------------------------------------------------------
#select features to perform training with random forest with k columns
# select the training dataset on k-features
newX_train = X_train.iloc[:, clf.feature_importances_.argsort()[::-1][:15]]
feature_select_improve = data.iloc[:, clf.feature_importances_.argsort()[::-1][:25]]
feature_select_improve['new_target']= data['new_target']
feature_select_improve.to_csv('feature_select_improve.csv', index=False)
# selrect the testing dataset on k-features
newX_test = X_test.iloc[:, clf.feature_importances_.argsort()[::-1][:15]]

#%%-----------------------------------------------------------------------
#perform training with random forest with k columns
# specify random forest classifier
clf_k_features = RandomForestClassifier(n_estimators=100)

# train the model
clf_k_features.fit(newX_train, y_train)

#%%-----------------------------------------------------------------------
#make predictions

# predicton on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

# prediction on test using k features
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)


# %%-----------------------------------------------------------------------
# calculate metrics gini model

print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print (mean_squared_error(y_test, y_pred))
#print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)

# calculate metrics entropy model
print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_k_features))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
print("\n")
#print("ROC_AUC : ", roc_auc_score(y_test,y_pred_k_features_score[:,1]) * 100)

# %%-----------------------------------------------------------------------
# confusion matrix for gini model
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
#df_cm.to_csv('RF_gini_cm.csv')
plt.show()

# %%-----------------------------------------------------------------------

# confusion matrix for entropy model

conf_matrix = confusion_matrix(y_test, y_pred_k_features)
class_names = data['new_target'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(25,25))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
plt.ylabel('True label',fontsize=100)
plt.xlabel('Predicted label',fontsize=100)
plt.tight_layout()
df_cm.to_csv('RF_gini_cm_improve.csv')
plt.show()
