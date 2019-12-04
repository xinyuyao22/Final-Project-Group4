import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import sys
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train_fea_eng.csv')
print (data.head())

X = data.drop(columns = (['target','first_active_month']))
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

id_train = X_train['card_id'].copy()
X_train.drop('card_id', axis = 1, inplace = True)
id_test = X_test['card_id'].copy()
X_test.drop('card_id', axis = 1, inplace = True)

ss = StandardScaler()
y_train = ss.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

nfolds = 10
folds = KFold(n_splits= nfolds, shuffle=True, random_state=15)

param = {'num_leaves': 50,
         'min_data_in_leaf': 30,
         'objective':'regression',
         'max_depth': 10,
         'learning_rate': 0.005,
         "min_child_samples": 100,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1}
feature_importance_df = np.zeros((X_train.shape[1], nfolds))
mvalid = np.zeros(len(X_train))
y_pred = np.zeros(len(X_test))

start = time.time()

X_train = pd.DataFrame(X_train,index=True)
y_train = pd.DataFrame(y_train,index=True)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):
    print('----')
    print("fold n°{}".format(fold_))

    x0, y0 = X_train.iloc[trn_idx], y_train[trn_idx]
    x1, y1 = X_train.iloc[val_idx], y_train[val_idx]

    trn_data = lgb.Dataset(x0, label=y0)
    val_data = lgb.Dataset(x1, label=y1)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data],
                    verbose_eval=500, early_stopping_rounds=150)
    mvalid[val_idx] = clf.predict(x1, num_iteration=clf.best_iteration)
    feature_importance_df[:, fold_] = clf.feature_importance()

    y_pred += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

ximp = pd.DataFrame()
ximp['feature'] = X_train.columns
ximp['importance'] = feature_importance_df.mean(axis = 1)

plt.figure(figsize=(14,14))
sns.barplot(x="importance",
            y="feature",
            data=ximp.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
from sklearn.metrics import r2_score

print('Mean_squared_error:', mean_squared_error(y_test, y_pred, multioutput='uniform_average'))
print('r2_score:', r2_score(y_test, y_pred, multioutput='variance_weighted'))