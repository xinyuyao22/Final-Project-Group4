import pandas as pd
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



data = pd.read_csv('train_fea_eng.csv',sep = ',')

print(data['target'].min())
print(data['target'].max())


bins = [-34,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,
       -14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
names = [-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,-16,-15,
       -14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
data['new_target'] = pd.cut(data['target'], bins, labels=names)

print(data.isnull().sum())

#category = pd.cut(data.target,bin)
#category = category.to_frame()
#category.columns = ['new_target']
#data['new_target'] = category


X = data.drop(columns = (['new_target','target','first_active_month','card_id']))
#Y = data[['new_target']]
Y = data.values[:, -1]
#lab_enc = preprocessing.LabelEncoder()
#training_scores_encoded = lab_enc.fit_transform(Y)
Y=Y.astype('int')
#Y=Y.astype('float')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

clf = GaussianNB()
clf.fit(X_train, y_train)

# predicton on test
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

#print (mean_squared_error(y_test, y_pred))

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
#print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['new_target'].unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()