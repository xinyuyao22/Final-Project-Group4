import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
data0 = pd.read_csv('newData_select.csv')
data = data0.sample(frac=0.01, random_state=23)

X = data[["purchase_amount_count_std", "auth_purchase_month_std"]]
y = data['new_target']
X = pd.DataFrame(X)
y = pd.DataFrame(y)
data = pd.merge(X, y, left_index=True, right_index=True, how='outer')
data.columns = ['x1', 'x2', 'y']
h = 1
x_min, x_max = data.x1.min() - 1, data.x1.max() + 1
y_min, y_max = data.x2.min() - 1, data.x2.max() + 1

sns.scatterplot(x=data.x1, y=data.x2, hue=data.y)
plt.title("Scatterplot before SVM")
plt.show()

X = data[['x1', 'x2']]
y = data.y
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3, random_state=13)
clf = SVC(C=6,kernel='rbf')
clf.fit(X_train,y_train)
y_pre = clf.predict(X)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

color = ["#F1948A", "#BB8FCE", "#85C1E9", "#58D68D", "#F5B041", "#CACFD2", "#5D6D7E"]
sns.scatterplot(data.x1, y=data.x2,hue=y_pre)
plt.contourf(xx, yy, Z, colors=color, alpha=0.2)
plt.title("Scatterplot after SVM")
plt.show()
print("accuracy:", accuracy_score(data.y,y_pre))


scores = []
for m in range(3,X_train.size):
    clf.fit(X_train[:m],y_train[:m])
    y_train_predict = clf.predict(X_train[:m])
    y_val_predict = clf.predict(X_val)
    scores.append(accuracy_score(y_train_predict,y_train[:m]))
plt.plot(range(3,X_train.size),scores,c='green', alpha=0.6)
plt.title("SVM Accuracy vs Sample")
plt.show()