
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# %%-----------------------------------------------------------------------
# Importing the required packages

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# %%-----------------------------------------------------------------------
#importing Dataset

data = pd.read_csv('train_fea_eng.csv')
bins = np.arange(-12.5, 12.5, 1)
names = np.arange(-12, 12, 1)
data['new_target'] = pd.cut(data['target'], bins, labels=names)

# %%-----------------------------------------------------------------------
# Data preprocessing

# look at first few rows
data.head()

# replace missing characters as NaN
data.replace('?', np.NaN, inplace=True)

# check the structure of mushroom_data
data.info()

# check the null values in each column
print(data.isnull().sum())

# check the summary of the mushroom_data
data.describe(include='all')

# replace categorical mushroom_data with the most frequent value in that column
data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))

# again check the null values in each column
data.isnull().sum()

# %%-----------------------------------------------------------------------
# One Hot Encoding the variables

# encoding the features using get dummies
X = data.drop(columns = (['new_target','target','first_active_month','card_id']))


X = X.values


# encoding the class with sklearn's LabelEncoder
Y = data['new_target']

class_le = LabelEncoder()

# fit and transform the class
y = class_le.fit_transform(Y)

# %%-----------------------------------------------------------------------
# split the mushroom_data

# Spliting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# %%-----------------------------------------------------------------------
# perform training

# creating the classifier object
clf = SVC(kernel="linear")

# performing training
clf.fit(X_train, y_train)

# %%-----------------------------------------------------------------------

# make predictions

# predicton on test
y_pred = clf.predict(X_test)


#%%-----------------------------------------------------------------------

# calculate metrics

print("\n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

#%%-----------------------------------------------------------------------
# function to display feature importance of the classifier

# here we will display top 20 features (top 10 max positive and negative coefficient values)

def coef_values(coef, names):
    imp = coef
    print(imp)
    imp,names = zip(*sorted(zip(imp.ravel(),names)))

    imp_pos_10 = imp[-10:]
    names_pos_10 = names[-10:]
    imp_neg_10 = imp[:10]
    names_neg_10 = names[:10]

    imp_top_20 = imp_neg_10+imp_pos_10
    names_top_20 =  names_neg_10+names_pos_10

    plt.barh(range(len(names_top_20)), imp_top_20, align='center')
    plt.yticks(range(len(names_top_20)), names_top_20)
    plt.show()

# get the column names
features_names = X_data.columns

# call the function
coef_values(clf.coef_, features_names)

#%%-----------------------------------------------------------------------

# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['new_target'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False,annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()


#%%-----------------------------------------------------------------------
# Plot ROC Area Under Curve

y_pred_proba = clf.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# print(fpr)
# print(tpr)
# print(auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
