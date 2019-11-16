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

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

new_trans = pd.read_csv('new_merchant_transactions.csv', parse_dates=['purchase_date'])
hist_trans = pd.read_csv('historical_transactions.csv', parse_dates=['purchase_date'])

def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

hist_trans = binarize(hist_trans)
new_trans = binarize(new_trans)

train = pd.read_csv('../Final Project/train.csv', parse_dates=["first_active_month"])
test = pd.read_csv('../Final Project/test.csv', parse_dates=["first_active_month"])

# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

# Missing values statistics
print(missing_values_table(hist_trans))
print(missing_values_table(new_trans))
print(missing_values_table(train))
print(missing_values_table(test))

for df in [hist_trans,new_trans]:
    df['category_2'].fillna(6.0,inplace=True)
    df['category_3'].fillna('D',inplace=True)
    df['merchant_id'].fillna('M_ID_na',inplace=True)

mean = (np.array(train['first_active_month'], dtype='datetime64[s]')
        .view('i8')
        .mean()
        .astype('datetime64[s]'))
test['first_active_month'].fillna(mean, inplace=True)

# Missing values statistics
print(missing_values_table(hist_trans))
print(missing_values_table(new_trans))
print(missing_values_table(train))
print(missing_values_table(test))

target = train['target']
plt.figure(figsize=(12, 5))
plt.hist(target.values, bins=200)
plt.title('Histogram target counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
# We can see that some of the loyalty values are far apart (less than -30) compared to others. Let us just get their count.
print((target<-30).sum(), 'entries have target<-30')
print("% of target<-30:", (target<-30).sum()/target.shape[0])

cnt_srs = train['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in train set")
plt.show()

cnt_srs = test['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in test set")
plt.show()

train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days

for df in [hist_trans,new_trans]:
    for col in ['category_2', 'category_3']:
        print(col, list(df[col].unique()))

hist_trans['month_diff'] = ((datetime.date(2018, 2, 1) - hist_trans['purchase_date'].dt.date).dt.days)//30
hist_trans['month_diff'] += hist_trans['month_lag']
new_trans['month_diff'] = ((datetime.date(2018, 2, 1) - new_trans['purchase_date'].dt.date).dt.days)//30
new_trans['month_diff'] += new_trans['month_lag']

train = pd.get_dummies(train, columns=['feature_1', 'feature_2', 'feature_3'])
test = pd.get_dummies(test, columns=['feature_1', 'feature_2', 'feature_3'])
hist_trans = pd.get_dummies(hist_trans, columns=['category_2', 'category_3'])
new_trans = pd.get_dummies(new_trans, columns=['category_2', 'category_3'])

hist_trans = reduce_mem_usage(hist_trans)
new_trans = reduce_mem_usage(new_trans)

agg_fun = {'authorized_flag': ['mean']}
auth_mean = hist_trans.groupby(['card_id']).agg(agg_fun)
auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
auth_mean.reset_index(inplace=True)

auth_trans = hist_trans[hist_trans['authorized_flag'] == 1]
hist_trans = hist_trans[hist_trans['authorized_flag'] == 0]

hist_trans['purchase_month'] = hist_trans['purchase_date'].dt.month
auth_trans['purchase_month'] = auth_trans['purchase_date'].dt.month
new_trans['purchase_month'] = new_trans['purchase_date'].dt.month

def aggregate_transactions(history):
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']). \
                                          astype(np.int64) * 1e-9

    agg_func = {
        'category_1': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum'],
        'purchase_month': ['mean', 'std'],
        'purchase_date': [np.ptp, 'min', 'max'],
        'month_lag': ['mean', 'std'],
        'month_diff': ['mean']
    }

    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)

    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))

    agg_history = pd.merge(df, agg_history, on='card_id', how='left')

    return agg_history

history = aggregate_transactions(hist_trans)
history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]
history.fillna(0, inplace=True)
print(missing_values_table(history))
print(history.head())

authorized = aggregate_transactions(auth_trans)
authorized.columns = ['auth_' + c if c != 'card_id' else c for c in authorized.columns]
print(missing_values_table(authorized))
print(authorized.head())

new = aggregate_transactions(new_trans)
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]
new.fillna(0, inplace=True)
print(missing_values_table(new))
print(new.head())

def aggregate_per_month(history):
    grouped = history.groupby(['card_id', 'month_lag'])

    agg_func = {
        'purchase_amount': ['count', 'mean', 'min', 'std'],
    }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)
    intermediate_group.fillna(0, inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    final_group.reset_index(inplace=True)

    return final_group


# ___________________________________________________________
final_group = aggregate_per_month(auth_trans)
print(missing_values_table(final_group))
print(final_group.head())

def successive_aggregates(df, field1, field2):
    t = df.groupby(['card_id', field1])[field2].mean()
    u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['min'])
    u.columns = [field1 + '_' + field2 + '_' + col for col in u.columns.values]
    u.reset_index(inplace=True)
    return u

additional_fields = successive_aggregates(new_trans, 'installments', 'purchase_amount')
additional_fields = additional_fields.merge(successive_aggregates(new_trans, 'city_id', 'purchase_amount'),
                                            on = 'card_id', how='left')

for df in [history, authorized, new, final_group, auth_mean, additional_fields]:
    train = pd.merge(train, df, on='card_id', how='left')
    test = pd.merge(test, df, on='card_id', how='left')

unimportant_features = ['hist_month_lag_std', 'hist_purchase_amount_max', 'hist_purchase_month_max', 'hist_purchase_month_min',
                        'hist_purchase_month_std', 'purchase_amount_mean_mean']

for df in [train, test]:
    for col in unimportant_features:
        df = df.drop(columns=[col], errors='ignore')

print(missing_values_table(train))
train.fillna(0, inplace=True)

train.to_csv('train_fea_eng_with_outliers.csv', index=False)
test.to_csv('test_fea_eng.csv', index=False)