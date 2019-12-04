from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")
data0 = pd.read_csv('newData_select.csv')
data = data0.sample(frac=0.01, random_state=23)

X = data[["month_lag_std", "auth_purchase_month_std"]]
y = data['new_target']
X = pd.DataFrame(X)
y = pd.DataFrame(y)
data = pd.merge(X, y, left_index=True, right_index=True, how='outer')
data.columns = ['x1', 'x2', 'y']
h = 1
x_min, x_max = data.x1.min() - 1, data.x1.max() + 1
y_min, y_max = data.x2.min() - 1, data.x2.max() + 1

sns.scatterplot(x=data.x1, y=data.x2, hue=data.y)
plt.title("Scatterplot before AGNES")
plt.show()

clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)

res = clustering.fit(X)

print("Number of samples for each cluster:")
print(pd.Series(clustering.labels_).value_counts())
print("Clustering results:")
print(confusion_matrix(data.y, clustering.labels_))

plt.figure()
d0 = X[clustering.labels_ == 0]
d1 = X[clustering.labels_ == 1]
d2 = X[clustering.labels_ == 2]
plt.plot(d0.iloc[:, 0], d0.iloc[:, 1], 'r.')
plt.plot(d1.iloc[:, 0], d1.iloc[:, 1], 'go')
plt.plot(d2.iloc[:, 0], d2.iloc[:, 1], 'b*')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("AGNES Clustering")
plt.show()
