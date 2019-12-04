import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train_fea_eng.csv')
print (data.head())

bins = np.arange(-12.5, 12.5, 1)
names = np.arange(-12, 12, 1)
data['new_target'] = pd.cut(data['target'], bins, labels=names)

X = data.drop(columns = (['new_target','target','first_active_month','card_id']))
y = data['new_target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

class_le = LabelEncoder()
y = class_le.fit_transform(y)
y_train = class_le.fit_transform(y_train)
y_test = class_le.fit_transform(y_test)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

clfs = {'dt': DecisionTreeClassifier(random_state=0)}

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe_clfs = {}

for name, clf in clfs.items():
    pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()),
                                ('clf', clf)])
param_grids = {}

param_grid = [{'clf__max_depth': [3, 6, 9], 'clf__min_samples_leaf': [1, 2, 3], 'clf__min_samples_split': [2, 3, 4]}]

param_grids['dt'] = param_grid

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# The list of [best_score_, best_params_, best_estimator_]
best_score_param_estimators = []

for name in pipe_clfs.keys():
    # GridSearchCV
    gs = GridSearchCV(estimator=pipe_clfs[name],
                      param_grid=param_grids[name],
                      scoring='accuracy',
                      n_jobs=1,
                      iid=False,
                      cv=StratifiedKFold(n_splits=10,
                                         shuffle=True,
                                         random_state=0))
    # Fit the pipeline
    gs = gs.fit(X, y)

    # Update best_score_param_estimators
    best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

print([best_score_param_estimators[0][0], best_score_param_estimators[0][1]], end='\n\n')

# Get the best estimator
#best_estimator = best_score_param_estimators[0][2]

#best_estimator_fit = best_estimator.fit(X_train, y_train)

# Predict the target value using the best estimator
#y_pred = best_estimator_fit.predict(X_test)
#%%-----------------------------------------------------------------------
# perform training with giniIndex.
# creating the classifier object
clf_gini = DecisionTreeClassifier(random_state=0, max_depth=best_score_param_estimators[0][1]['clf__max_depth'],
                                  min_samples_split=best_score_param_estimators[0][1]['clf__min_samples_split'],
                                  min_samples_leaf=best_score_param_estimators[0][1]['clf__min_samples_leaf'])

# performing training
clf_gini.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
# make predictions
# predicton on test using gini
y_pred = clf_gini.predict(X_test)
#%%-----------------------------------------------------------------------
# calculate metrics gini model
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")
print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)
print('Mean_squared_error:', mean_squared_error(y_test, y_pred))
print("\n")
print ('-'*80 + '\n')
#%%-----------------------------------------------------------------------
# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = names
class_names = class_names.astype('str')
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
df_cm.to_csv('dt_cm.csv')

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 4}, yticklabels=df_cm.columns,
                 xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=10)
plt.xlabel('Predicted label',fontsize=10)
plt.tight_layout()
plt.savefig('con_dt.pdf')
plt.show()
#%%-----------------------------------------------------------------------
# display decision tree
dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=class_names, feature_names=X.columns,
                           out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_gini.pdf")
webbrowser.open_new(r'decision_tree_gini.pdf')