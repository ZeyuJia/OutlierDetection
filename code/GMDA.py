import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import VotingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score, \
    confusion_matrix, roc_curve, roc_auc_score


data = pd.read_csv('Statistical_Learning/data/creditcard.csv')
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

class GMM():

    def __init__(self, n_normal=3, n_anomaly=4, cov_type='full', random_state=42):
        self.normal = GaussianMixture(n_components=n_normal, covariance_type=cov_type,
                                      n_init=5, max_iter=500, random_state=random_state, tol=1e-5)
        self.anomaly = GaussianMixture(n_components=n_anomaly, covariance_type=cov_type,
                                       n_init=5, max_iter=500, random_state=random_state, tol=1e-5)

    def fit(self, train_normal, train_anomaly):
        self.normal.fit(train_normal.drop(columns=['Class']))
        self.anomaly.fit(train_anomaly.drop(columns=['Class']))

    def predict(self, sets, thre):
        return (thre + self.normal.score_samples(sets.drop(columns=['Class'])) \
                <= self.anomaly.score_samples(sets.drop(columns=['Class']))).astype(np.int32)

    def Valid(self, sets, tune, draw=0):
        score = []
        for p in tune:
            pred = self.predict(sets, p)
            recall = recall_score(y_true=sets['Class'].values, y_pred=pred)
            precision = precision_score(y_true=sets['Class'].values, y_pred=pred)
            fbeta = fbeta_score(y_true=sets['Class'].values, y_pred=pred, beta=2)
            score.append([recall, precision, fbeta])
        score = np.array(score)
        if draw:
            plt.plot(tune, score[:, 0], label='$Recall$')
            plt.plot(tune, score[:, 1], label='$Precision$')
            plt.plot(tune, score[:, 2], label='$F_2$')
            plt.ylabel('Score')
            plt.xlabel('Threshold')
            plt.legend(loc='best')
            plt.show()
            pred = self.predict(sets, self.threshold)
            result(sets, pred)
        self.threshold = tune[score[:, 2].argmax()]

    def Test(self, sets):
        pred = self.predict(sets, self.threshold)
        result(sets, pred)


def feature(self, sets):
    feature_normal = np.array([])

    l = len(self.normal.means_)
    for i in range(l):
        x = np.array(sets.drop(columns=['Class']) - self.normal.means_[i])
        x = np.sum(x.dot(np.linalg.inv(self.normal.covariances_[i])) * x, axis=1)
        feature_normal = np.append(feature_normal, x, axis=0)
    feature_normal = feature_normal.reshape(l, -1)
    feature_anomaly = np.array([])

    l = len(self.anomaly.means_)
    for i in range(l):
        x = np.array(sets.drop(columns=['Class']) - self.anomaly.means_[i])
        x = np.sum(x.dot(np.linalg.inv(self.anomaly.covariances_[i])) * x, axis=1)
        feature_anomaly = np.append(feature_anomaly, x, axis=0)
    feature_anomaly = feature_anomaly.reshape(l, -1)

    normal_score = self.normal.score_samples(sets.drop(columns=['Class'])).reshape(1, -1)
    anomaly_score = self.anomaly.score_samples(sets.drop(columns=['Class'])).reshape(1, -1)

    features = normal_score - anomaly_score
    features = np.concatenate((features, feature_normal, feature_anomaly), axis=0)

    for i in range(len(feature_normal)):
        for j in range(len(feature_anomaly)):
            features = np.append(features, np.array([feature_normal[i] - feature_anomaly[j]]), axis=0)

    return features


def result_all(predictor):
    sets = train
    predict = predictor.predict(feature(gmm, sets).T)
    result(sets, predict)

    sets = valid
    predict = predictor.predict(feature(gmm, sets).T)
    result(sets, predict)

    sets = test
    time_start = time()
    predict = predictor.predict(feature(gmm, sets).T)
    time_end = time()
    result(sets, predict)

    return (time_end - time_start)


train_start = time()
gmm = GMM(n_anomaly=4)
gmm.fit(train_normal, train_anomaly)
train_end = time()

gmm.Valid(valid, np.linspace(50, 150, 101))

print('GMDA ', end='')
gmm.Test(train)
gmm.Test(valid)
test_start = time()
gmm.Test(test)
test_end = time()
print('& {0:.3f} & {1:.3f} \\\\\\hline'.format(train_end - train_start, test_end - test_start))


print('GM-dtc ', end='')
train_start = time()
dtc = DecisionTreeClassifier(criterion='entropy', max_depth=5).fit(feature(gmm, train).T, train['Class'])
train_end = time()
test_time = result_all(dtc)
print('& {0:.3f} & {1:.3f} \\\\\\hline'.format(train_end - train_start, test_time))


bgc = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=5),
                        bootstrap_features=False, n_estimators=11)
print('GM-bagging ', end='')
train_start = time()
bgc.fit(feature(gmm, train).T, train['Class'])
train_end = time()
test_time = result_all(bgc)
print('& {0:.3f} & {1:.3f} \\\\\\hline'.format(train_end - train_start, test_time))


estimators = [
              ('dtc1',DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight={0:1, 1:0.01})),
              ('dtc2',DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight={0:1, 1:0.02})),
              ('dtc3',DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight={0:1, 1:0.05})),
              ('dtc4',DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight={0:1, 1:0.1})),
              ('dtc5',DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight={0:1, 1:1})),
              ('dtc6',DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight={0:1, 1:10})),
              ('dtc7',DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight={0:1, 1:20})),
              ('dtc8',DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight={0:1, 1:50})),
              ('dtc9',DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight={0:1, 1:100})),
             ]
vtc = VotingClassifier(estimators=estimators, voting='soft')#, weights=[1, 1.1, 1.1, 1.1, 1.1, 1])

print('GM-voting ', end='')
train_start = time()
bgc.fit(feature(gmm, train).T, train['Class'])
train_end = time()
test_time = result_all(bgc)
print('& {0:.3f} & {1:.3f} \\\\\\hline'.format(train_end - train_start, test_time))


x = valid
train_start = time()
tune = np.linspace(-800, -700, 101)
score = []
for p in tune:
    predict_gmm = (gmm.normal.score_samples(x.drop(columns=['Class'])) <= p).astype(np.int32)
    predict = predict_gmm
    recall_gmm = recall_score(y_true=x['Class'].values, y_pred=predict)
    precision_gmm = precision_score(y_true=x['Class'].values, y_pred=predict)
    fbeta_gmm = fbeta_score(y_true=x['Class'].values, y_pred=predict, beta=1)
    score.append([recall_gmm, precision_gmm, fbeta_gmm])
train_end = time()

score = np.array(score)

p = tune[score[:, 2].argmax()]
print('GMDA-n ', end='')
predict = (gmm.normal.score_samples(train.drop(columns=['Class'])) <= p).astype(np.int32)
result(train, predict)

predict = (gmm.normal.score_samples(valid.drop(columns=['Class'])) <= p).astype(np.int32)
result(valid, predict)

test_start = time()
predict = (gmm.normal.score_samples(test.drop(columns=['Class'])) <= p).astype(np.int32)
result(test, predict)
test_end = time()
print('& {0:.3f} & {1:.3f} \\\\\\hline'.format(train_end - train_start, test_end - test_start))
