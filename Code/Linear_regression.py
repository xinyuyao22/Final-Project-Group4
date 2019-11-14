import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 500)

train = pd.read_csv('train_fea_eng.csv', parse_dates=["first_active_month"])
# test = pd.read_csv('test_fea_eng.csv', parse_dates=["first_active_month"])

# Check NA
print(train.isna().sum().sum())
X = train.drop(columns=['target', 'first_active_month', 'card_id'])
y = train['target'].values

from sklearn.model_selection import train_test_split

# Divide the data into training and testing (with test_size=0.3 and random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
y_train = ss.fit_transform(y_train.reshape(-1, 1)).reshape(-1) # .reshape(-1) transform into array

from sklearn.base import BaseEstimator, RegressorMixin

class MyFastLinearRegression(BaseEstimator, RegressorMixin):
    """The fast linear regression model (implemented using numpy array)"""

    def __init__(self, n_iter=100, eta=10 ** -2, C=1, random_state=0):
        # The number of iterations
        self.n_iter = n_iter

        # The learning rate
        self.eta = eta

        # The regularization parameter
        self.C = C

        # The random state
        self.random_state = random_state

        # The cost
        self.cost = None

    def fit(self, X, y):
        """
        The fit function

        Parameters
        ----------
        X : the feature matrix
        y : the target vector
        """

        # Initialize the cost
        self.cost = []

        # The random number generator
        self.rgen = np.random.RandomState(seed=self.random_state)

        # Initialize the weight for features x0 (the dummy feature), x1, x2, ..., xn
        self.w = self.rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)

        # For each iteration
        for _ in range(self.n_iter):
            # Get the net_input
            net_input = self.net_input(X)

            # Get the errors
            errors = y - net_input

            # Get the mean squared error (mse)
            mse = (errors ** 2).sum() / X.shape[0]

            # Update the cost
            self.cost.append(mse)

            # Update the weight of features x1, x2, ..., xn
            self.w[1:] += self.eta * (2 * np.matmul(X.T, errors) / X.shape[0] - self.C * np.sign(self.w[1:]))

            # Update the weight of the dummy feature, x0
            self.w[0] += self.eta * 2 * errors.sum() / X.shape[0]

    def net_input(self, X):
        """
        Get the net input

        Parameters
        ----------
        X : the feature matrix

        Returns
        ----------
        The net input

        """
        return np.matmul(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        """
        The predict function

        Parameters
        ----------
        X : the feature matrix

        Returns
        ----------
        The predicted value of the target
        """
        return self.net_input(X)

from sklearn.preprocessing import StandardScaler

estimators = {'mylr': MyFastLinearRegression(random_state=0)}
from sklearn.pipeline import Pipeline

pipe_estimators = {}

for name, estimator in estimators.items():
    pipe_estimators[name] = Pipeline([('StandardScaler', StandardScaler()),
                                      ('estimator', estimator)])

from sklearn.model_selection import GridSearchCV

param_grids = {}

eta_range = [10 ** i for i in range(-4, 0)]
C_range = [10 ** i for i in range(-7, 2)]

param_grid = [{'estimator__eta': eta_range,
               'estimator__C': C_range}]

param_grids['mylr'] = param_grid

# The list of [best_score_, best_params_, best_estimator_]
best_score_param_estimators = []

for name in pipe_estimators.keys():
    # GridSearchCV
    gs = GridSearchCV(estimator=pipe_estimators[name],
                      param_grid=param_grids[name],
                      scoring='neg_mean_squared_error',
                      n_jobs=-1,
                      iid=False,
                      cv=KFold(n_splits=10,
                               random_state=0),
                      return_train_score=True)

    gs = gs.fit(X_train, y_train)

    # Update best_score_param_estimators
    best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score',
                         'std_test_score',
                         'mean_train_score',
                         'std_train_score',
                         'mean_fit_time',
                         'std_fit_time',
                         'mean_score_time',
                         'std_score_time']

    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

    # Write cv_results file
    cv_results.to_csv(path_or_buf=name + '_cv_results.csv', index=False)

best_score, best_params, best_estimator = best_score_param_estimators[0]

print('%-15s' % 'best_score:', best_score)
print('%-15s' % 'best_estimator:'.format(20), type(best_estimator))
print('%-15s' % 'best_params:'.format(20), best_params, end='\n\n')

# Get the best estimator
best_estimator = best_score_param_estimators[0][2]

# Predict the target value using the best estimator
y_pred = best_estimator.predict(X_test)

# Transform the predicted target value back to the original scale
y_pred = ss.inverse_transform(y_pred)

from sklearn.metrics import mean_squared_error

print('Mean_squared_error', mean_squared_error(y_test, y_pred))