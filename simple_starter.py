import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


train = pd.read_csv('data/act_train.csv', parse_dates=['date'], dtype={'people_id': np.str, 'activity_id': np.str})
test = pd.read_csv('data/act_test.csv', parse_dates=['date'], dtype={'people_id': np.str, 'activity_id': np.str})
ppl = pd.read_csv('data/people.csv', parse_dates=['date'])

df_train = pd.merge(train, ppl, on='people_id')
df_test = pd.merge(test, ppl, on='people_id')
del train, test, ppl

from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
import zipfile
import time
import shutil
from sklearn.metrics import log_loss
import datetime
from sklearn.metrics import roc_auc_score

random.seed(2016)

def run_xgb(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 3
    subsample = 0.7
    colsample_bytree = 0.7
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta,
                                                                                                max_depth,
                                                                                                subsample,
                                                                                                colsample_bytree))
    params = {
        "learning_rate": 0.5,
        "n_estimators": 1000,
        "objective": "binary:logistic",
        # "num_class": len(np.unique(train[target])),
        "booster": "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }

    num_boost_round = 500
    early_stopping_rounds = 50
    test_size = 0.3

    # from sklearn import preprocessing
    # scaler = preprocessing.StandardScaler().fit(train[features])
    # train[features] = scaler.transform(train[features])
    # test[features] = scaler.transform(test[features])

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]

    print(X_train[features].head())
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)
    print(check)
    score = roc_auc_score(y_valid.tolist(), check)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))

    return test_prediction.tolist(), score


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('activity_id,outcome\n')
    total = 0
    test_val = test['activity_id'].values
    for i in range(len(test_val)):
        str1 = str(test_val[i]) + ',' + str(prediction[i])
        # for j in range(12):
        #     str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()

def map_column(table, f):
    table[f] = table[f].fillna('None')
    print(np.unique(table[f]))
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table

print(df_train.head())
print(df_train.columns)

# features['date_x_prob'] = df_train.groupby('date_x')['outcome'].transform('mean')
# features['date_y_prob'] = df_train.groupby('date_y')['outcome'].transform('mean')
# features['date_x_count'] = df_train.groupby('date_x')['outcome'].transform('count')
# features['date_y_count'] = df_train.groupby('date_y')['outcome'].transform('count')

columns = ['date_x', 'activity_category', 'char_1_x',
       'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x',
       'char_8_x', 'char_9_x', 'char_1_y',
       'char_2_y', 'date_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y',
       'char_7_y', 'char_8_y', 'char_9_y', 'char_10_y', 'char_11', 'char_12',
       'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18',
       'char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24',
       'char_25', 'char_26', 'char_27', 'char_28', 'char_29', 'char_30',
       'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36',
       'char_37', 'char_38'
           ]

for column in columns:
    print(column, df_train[column].dtype)
    if df_train[column].dtype == bool:
        df_train[column] = df_train[column].map({True: 1, False: 0})
        df_test[column] = df_test[column].map({True: 1, False: 0})

    if df_train[column].dtype == object:
        # df_train = pd.concat([df_train,
        #                       pd.get_dummies(df_train[column], prefix=column)],
        #                      axis=1)
        df_train = map_column(df_train, column)
        df_test = map_column(df_test, column)
        continue
    if 'datetime64' in str(df_train[column].dtype):
        df_train['year'] = pd.DatetimeIndex(df_train[column]).year
        df_train['month'] = pd.DatetimeIndex(df_train[column]).month
        df_train['day'] = pd.DatetimeIndex(df_train[column]).weekday
        df_train.drop(column, axis=1, inplace=True)

        df_test['year'] = pd.DatetimeIndex(df_test[column]).year
        df_test['month'] = pd.DatetimeIndex(df_test[column]).month
        df_test['day'] = pd.DatetimeIndex(df_test[column]).weekday
        df_test.drop(column, axis=1, inplace=True)

features = []
for column in df_train.columns:
    if column not in columns + ['activity_id', 'outcome']:
        df_train.drop(column, axis=1, inplace=True)
        df_test.drop(column, axis=1, inplace=True)

features = df_train.columns - ['activity_id', 'outcome']
print(features)
for column in df_train.columns:
    print(column, df_train[column].dtype)
print(features)
print(df_train.head())

test_predictions, score = run_xgb(df_train, df_test, features, 'outcome', 25)
#test_predictions = run_nn(train, test, features, 'group', 25)
#test_predictions = run_knn(train, test, features, 'group', n_neighbors=3, weights='distance')
#print("LS: {}".format(round(score, 5)))
create_submission(0.0, df_test, test_predictions)

# pd.get_dummies(data['Sex'], prefix='Sex').head(5)


# for d in ['date_x', 'date_y']:
#     print('Start of ' + d + ': ' + str(df_train[d].min().date()))
#     print('  End of ' + d + ': ' + str(df_train[d].max().date()))
#     print('Range of ' + d + ': ' + str(df_train[d].max() - df_train[d].min()) + '\n')
#
# date_x = pd.DataFrame()
# date_x['Class probability'] = df_train.groupby('date_x')['outcome'].mean()
# date_x['Frequency'] = df_train.groupby('date_x')['outcome'].size()
# # date_x.plot(secondary_y='Frequency', figsize=(20, 10))
#
#
# date_y = pd.DataFrame()
# date_y['Class probability'] = df_train.groupby('date_y')['outcome'].mean()
# date_y['Frequency'] = df_train.groupby('date_y')['outcome'].size()
# # We need to split it into multiple graphs since the time-scale is too long to show well on one graph
# i = int(len(date_y) / 3)
# # date_y[:i].plot(secondary_y='Frequency', figsize=(20, 5), title='date_y Year 1')
# # date_y[i:2*i].plot(secondary_y='Frequency', figsize=(20, 5), title='date_y Year 2')
# # date_y[2*i:].plot(secondary_y='Frequency', figsize=(20, 5), title='date_y Year 3')
#
# date_x_freq = pd.DataFrame()
# date_x_freq['Training set'] = df_train.groupby('date_x')['activity_id'].count()
# date_x_freq['Testing set'] = df_test.groupby('date_x')['activity_id'].count()
# date_x_freq.plot(secondary_y='Testing set', figsize=(20, 8),
#                  title='Comparison of date_x distribution between training/testing set')
# date_y_freq = pd.DataFrame()
# date_y_freq['Training set'] = df_train.groupby('date_y')['activity_id'].count()
# date_y_freq['Testing set'] = df_test.groupby('date_y')['activity_id'].count()
# date_y_freq[:i].plot(secondary_y='Testing set', figsize=(20, 8),
#                  title='Comparison of date_y distribution between training/testing set (first year)')
# date_y_freq[2*i:].plot(secondary_y='Testing set', figsize=(20, 8),
#                  title='Comparison of date_y distribution between training/testing set (last year)')
#
# print('Correlation of date_x distribution in training/testing sets: ' + str(np.corrcoef(date_x_freq.T)[0,1]))
# print('Correlation of date_y distribution in training/testing sets: ' + str(np.corrcoef(date_y_freq.fillna(0).T)[0,1]))
#
# print('date_y correlation in year 1: ' + str(np.corrcoef(date_y_freq[:i].fillna(0).T)[0,1]))
# print('date_y correlation in year 2: ' + str(np.corrcoef(date_y_freq[i:2*i].fillna(0).T)[0,1]))
# print('date_y correlation in year 3: ' + str(np.corrcoef(date_y_freq[2*i:].fillna(0).T)[0,1]))
#
# from sklearn.metrics import roc_auc_score
# features = pd.DataFrame()
# features['date_x_prob'] = df_train.groupby('date_x')['outcome'].transform('mean')
# features['date_y_prob'] = df_train.groupby('date_y')['outcome'].transform('mean')
# features['date_x_count'] = df_train.groupby('date_x')['outcome'].transform('count')
# features['date_y_count'] = df_train.groupby('date_y')['outcome'].transform('count')
# _=[print(f.ljust(12) + ' AUC: ' + str(round(roc_auc_score(df_train['outcome'], features[f]), 6))) for f in features.columns]


