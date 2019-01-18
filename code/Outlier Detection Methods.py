import numpy as np
import pandas as pd
from time import time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score


data = pd.read_csv('Project/Statistical_Learning/data/creditcard.csv')
x = np.array(data.loc[1:len(data)-1, 'Time'])
y = np.array(data.loc[0:len(data)-2, 'Time'])
data.loc[1:, 'Time'] = x-y
normal = data[data['Class']==0]
anomaly = data[data['Class']==1]


train_normal, test_normal = train_test_split(normal, test_size=0.4, random_state=42)
valid_normal, test_normal = train_test_split(test_normal, test_size=1/2, random_state=42)
train_anomaly, test_anomaly = train_test_split(anomaly, test_size=0.4, random_state=42)
valid_anomaly, test_anomaly = train_test_split(test_anomaly, test_size=1/2, random_state=42)

for x in [train_normal, valid_normal, test_normal, train_anomaly, valid_anomaly, test_anomaly]:
    x.reset_index(drop=True, inplace=True)

print('Normal Train:', train_normal.shape,
      'Normal Valid:', valid_normal.shape,
      'Normal Test:', test_normal.shape)
print('Anomaly Train:', train_anomaly.shape,
      'Anomaly Valid:', valid_anomaly.shape,
      'Anomaly Test:', test_anomaly.shape)


train = train_normal.append(train_anomaly).sample(frac=1, random_state=42).reset_index(drop=True)
valid = valid_normal.append(valid_anomaly).sample(frac=1, random_state=42).reset_index(drop=True)
test = test_normal.append(test_anomaly).sample(frac=1, random_state=42).reset_index(drop=True)


def result(sets, predict):
    recall = recall_score(y_true=sets['Class'].values, y_pred=predict)
    precision = precision_score(y_true=sets['Class'].values, y_pred=predict)
    fbeta = fbeta_score(y_true=sets['Class'].values, y_pred=predict, beta=1.5)

    print('& {0:.3f} & {1:.3f} & {2:.3f} '.format(100 * recall, 100 * precision, 100 * fbeta), end='')


train_start = time()
robust_cov = MinCovDet().fit(train)
train_end = time()
p = 10**6+4*10**4
print('RCE ', end='')
predict = (robust_cov.mahalanobis(train) > p).astype(np.int32)
result(train, predict)
predict = (robust_cov.mahalanobis(valid) > p).astype(np.int32)
result(valid, predict)
test_start = time()
predict = (robust_cov.mahalanobis(test) > p).astype(np.int32)
result(test, predict)
test_end = time()
print('& {0:.2f} & {1:.2f} \\\\'.format(train_end - train_start, test_end - test_start))


train_start = time()
lof = LocalOutlierFactor(n_neighbors=50, novelty=True, contamination=0.0008).fit(train_normal.drop(columns=['Class']))
train_end = time()
print('LOF ', end='')
x = train
predict = (1-lof.predict(x.drop(columns=['Class'])))/2
result(train, predict)
x = valid
predict = (1-lof.predict(x.drop(columns=['Class'])))/2
result(valid, predict)
x = test
test_start = time()
predict = (1-lof.predict(x.drop(columns=['Class'])))/2
result(test, predict)
test_end = time()
print('& {0:.2f} & {1:.2f} \\\\'.format(train_end - train_start, test_end - test_start))


time_start = time()
ilf = IsolationForest(n_estimators=250, contamination=0.005, bootstrap=False,
                      behaviour='new').fit(train_normal.drop(columns=['Class']))
time_end = time()
print('ILF ', end='')
x = train
predict = (1-ilf.predict(x.drop(columns=['Class'])))/2
result(train, predict)
x = valid
predict = (1-ilf.predict(x.drop(columns=['Class'])))/2
result(valid, predict)
x = test
test_start = time()
predict = (1-ilf.predict(x.drop(columns=['Class'])))/2
result(test, predict)
test_end = time()
print('& {0:.2f} & {1:.2f} \\\\'.format(train_end - train_start, test_end - test_start))
