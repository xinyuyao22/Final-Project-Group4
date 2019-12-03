from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


warnings.filterwarnings("ignore")
data0 = pd.read_csv('newData_select.csv')
data = data0.sample(frac=0.01, random_state=23)

X = data[["auth_purchase_month_mean", "auth_purchase_month_std"]]
y = data['new_target']
X = pd.DataFrame(X)
y = pd.DataFrame(y)
data = pd.merge(X, y, left_index=True, right_index=True, how='outer')
data.columns = ['x1', 'x2', 'y']
h = 1
x_min, x_max = data.x1.min() - 1, data.x1.max() + 1
y_min, y_max = data.x2.min() - 1, data.x2.max() + 1

sns.scatterplot(x=data.x1, y=data.x2, hue=data.y)
plt.title("Scatterplot before Kmeans")
plt.show()

estimator = KMeans(n_clusters=3)
estimator.fit(X)
label_pred = estimator.labels_

x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0.iloc[:, 0], x0.iloc[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1.iloc[:, 0], x1.iloc[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2.iloc[:, 0], x2.iloc[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title("Kmeans Clustering")
plt.legend(loc=2)
plt.show()
