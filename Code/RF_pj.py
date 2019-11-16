import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

data = pd.read_csv('train_fea_eng.csv',sep = ',')

#print(data.isnull().sum())
data.dropna(axis=1, inplace=True)
bins = [-34,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,
       -14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
names = [-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,
       -14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
data['new_target'] = pd.cut(data['target'], bins, labels=names)


X = data.drop(columns = (['new_target','target','first_active_month','card_id']))
Y = data[['new_target']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
clf = RandomForestClassifier(n_estimators=100)
#lab_enc = preprocessing.LabelEncoder()
#training_scores_encoded = lab_enc.fit_transform(Y)
#clf.fit(X, training_scores_encoded)
clf.fit(X_train, y_train)

importances = clf.feature_importances_
f_importances = pd.Series(importances, data.drop(columns = (['new_target','target','first_active_month','card_id'])))
f_importances.sort_values(ascending=False, inplace=True)
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)
plt.tight_layout()
plt.show()

newX_train = X_train[:, clf.feature_importances_.argsort()[::-1][:15]]
newX_test = X_test[:, clf.feature_importances_.argsort()[::-1][:15]]

clf_k_features = RandomForestClassifier(n_estimators=100)
clf_k_features.fit(newX_train, y_train)

y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)