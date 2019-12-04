import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
data0 = pd.read_csv('newData_select.csv')
data1 = data0.sample(frac=0.1, random_state=23)
X1 = data1.drop(columns=(['new_target']))
Y1 = data1['new_target']
class_le = LabelEncoder()
y1 = class_le.fit_transform(Y1)


def KNN(K):
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=100, stratify=y1)
    stdsc = StandardScaler()
    stdsc.fit(X_train)
    X_train_std = stdsc.transform(X_train)
    X_test_std = stdsc.transform(X_test)
    clf = KNeighborsClassifier(n_neighbors=K)
    clf.fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)

    print("\n")
    print("Classification Report: ")
    print(classification_report(y_test, y_pred))
    print("\n")
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    print("\n")

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_names = data1['new_target'].unique()
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

    plt.figure(figsize=(25, 25))
    hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20},
                     yticklabels=df_cm.columns, xticklabels=df_cm.columns)
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=50)
    plt.ylabel('True label', fontsize=60)
    plt.xlabel('Predicted label', fontsize=60)
    plt.title("KNN Heatmap (K=%d)" % K, fontsize=60)
    plt.tight_layout()
    plt.show()


for i in range(1, 10, 5):
    KNN(i)

k_range = range(1, 50, 3)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(k)

    scores = cross_val_score(knn, X1, y1, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

result = dict(zip(k_range, k_scores))
print(result)

plt.plot(k_range, k_scores)
plt.scatter(k_range, k_scores)
plt.title("KNN vs Accuracy")
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross_Validation Accuracy')
plt.show()
