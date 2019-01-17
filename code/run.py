import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

def print_data(model, train, valid, test):
	data = []
	predict_model = model.predict(train.drop(columns=['Class']))
	recall_model = recall_score(y_true=train['Class'].values, y_pred=predict_model)
	precision_model = precision_score(y_true=train['Class'].values, y_pred=predict_model)
	fbeta_model = fbeta_score(y_true=train['Class'].values, y_pred=predict_model, beta=1.5)

	data.append(recall_model)
	data.append(precision_model)
	data.append(fbeta_model)

	predict_model = model.predict(valid.drop(columns=['Class']))
	recall_model = recall_score(y_true=valid['Class'].values, y_pred=predict_model)
	precision_model = precision_score(y_true=valid['Class'].values, y_pred=predict_model)
	fbeta_model = fbeta_score(y_true=valid['Class'].values, y_pred=predict_model, beta=1.5)

	data.append(recall_model)
	data.append(precision_model)
	data.append(fbeta_model)

	start = time.time()
	predict_model = model.predict(test.drop(columns=['Class']))
	stop = time.time()
	recall_model = recall_score(y_true=test['Class'].values, y_pred=predict_model)
	precision_model = precision_score(y_true=test['Class'].values, y_pred=predict_model)
	fbeta_model = fbeta_score(y_true=test['Class'].values, y_pred=predict_model, beta=1.5)

	data.append(recall_model)
	data.append(precision_model)
	data.append(fbeta_model)

	t = stop - start

	return data, t


def run(model, train, valid, test):
	train_1 = train.drop(columns=['Class'])
	train_2 = train['Class']
	m = model
	start = time.time()
	m.fit(train_1, train_2)
	stop = time.time()
	t1 = stop - start
	data, t2 = print_data(m, train, valid, test)
	return data, t1, t2


def load_data(random_state = 42):
	data = pd.read_csv('../data/creditcard.csv')
	x = np.array(data.loc[1:len(data)-1, 'Time'])
	y = np.array(data.loc[0:len(data)-2, 'Time'])
	data.loc[1:, 'Time'] = x-y
	normal = data[data['Class']==0]
	anomaly = data[data['Class']==1]

	train_normal, test_normal = train_test_split(normal, test_size=0.4, random_state=random_state)
	valid_normal, test_normal = train_test_split(test_normal, test_size=0.5, random_state=random_state)
	train_anomaly, test_anomaly = train_test_split(anomaly, test_size=0.4, random_state=random_state)
	valid_anomaly, test_anomaly = train_test_split(test_anomaly, test_size=0.5, random_state=random_state)

	for x in [train_normal, valid_normal, test_normal, train_anomaly, valid_anomaly, test_anomaly]:
	    x.reset_index(drop=True, inplace=True)

	print('Normal Train:', train_normal.shape, 
	      'Normal Valid:', valid_normal.shape, 
	      'Normal Test:', test_normal.shape)
	print('Anomaly Train:', train_anomaly.shape, 
	      'Anomaly Valid:', valid_anomaly.shape, 
	      'Anomaly Test:', test_anomaly.shape)

	train = train_normal.append(train_anomaly).sample(frac=1, random_state=random_state).reset_index(drop=True)
	valid = valid_normal.append(valid_anomaly).sample(frac=1, random_state=random_state).reset_index(drop=True)
	test = test_normal.append(test_anomaly).sample(frac=1, random_state=random_state).reset_index(drop=True)

	return train, valid, test


train, valid, test = load_data()

logistic = LogisticRegression(solver='newton-cg', multi_class='multinomial')
gnb = GaussianNB()
tree = DecisionTreeClassifier(criterion='gini', max_depth=6)
# SVM = SVC(kernel='linear', C=0.4)
lda = LinearDiscriminantAnalysis()
lda_bagging = BaggingClassifier(LinearDiscriminantAnalysis(), n_estimators=5)
qda = QuadraticDiscriminantAnalysis()
qda_bagging = BaggingClassifier(QuadraticDiscriminantAnalysis(), n_estimators=11)
rfc = RandomForestClassifier()
nn = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(5, 4, 3, 3), random_state=1)
xg = XGBClassifier(max_depth=4, reg_lambda=0.5)

model_set = {'Logistic Regression': logistic,
			 'Naive Bayes': gnb,
			 'Decision Tree': tree,
			 'Linear Discriminant Analysis': lda,
			 'LDA with Bagging': lda_bagging,
			 'Quadratic Discriminant Analysis': qda,
			 'QDA with Bagging': qda_bagging,
			 'Random Forest': rfc,
			 'Neural Network': nn,
			 'XGBoost': xg}

f = open('Supervised.txt', 'w')
print('\\hline', file=f)
for item in model_set.keys():
	print(item, end=' & ', file=f)
	data, t1, t2 = run(model_set[item], train, valid, test)
	for i in data:
		print('%.1f'%(100*i), end=' & ', file=f)
	print('%.3f & %.3f'%(t1, t2), end='\\\\\n', file=f)
	print('\\hline', file=f)


