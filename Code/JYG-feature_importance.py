import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('train_fea_eng.csv')
bins = np.arange(-12.5, 12.5, 2)
names = np.arange(-12, 12, 2)
data['new_target'] = pd.cut(data['target'], bins, labels=names)

X = data.drop(columns=(['new_target', 'target', 'first_active_month', 'card_id']))
Y = data['new_target']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

importances = clf.feature_importances_
f_importances = pd.Series(importances,
                          data.drop(columns=(['new_target', 'target', 'first_active_month', 'card_id'])).columns)
f_importances.sort_values(ascending=False, inplace=True)
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)
plt.tight_layout()
plt.title("Feature Importance", fontsize=30)
plt.show()

select = clf.feature_importances_.argsort()[::-1][:25]
feature_select = data.iloc[:, select]
feature_select['new_target'] = data['new_target']
feature_select.to_csv('feature_select.csv', index=False)

data2 = pd.read_csv('feature_select.csv')
data2 = data2.sample(frac=0.3, random_state=66)
data2.to_csv('newData_select.csv', index=False)