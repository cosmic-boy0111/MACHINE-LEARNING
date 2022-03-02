from numpy.testing._private.utils import print_assert_equal
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 

import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold , StratifiedKFold
from sklearn.model_selection import cross_val_score


def get_score(model,X_train, X_test, y_train, y_test): 
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)


digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# lr = LogisticRegression()
# lr.fit(X_train,y_train)
# print('logistic : ', lr.score(X_test,y_test))

# svm = SVC(C=10)
# svm.fit(X_train,y_train)
# print('svm : ', svm.score(X_test,y_test))

# rf = RandomForestClassifier(n_estimators=100)
# rf.fit(X_train,y_train)
# print('rf : ', rf.score(X_test,y_test))


########### k fold ###########3

kf = KFold(n_splits=3)

folds = StratifiedKFold(n_splits=3)

score_logistic = []
score_svm = []
score_rf = []

for train_index, test_index in folds.split(digits.data,digits.target): 
    X_train,X_test,y_train,y_test = digits.data[train_index],digits.data[test_index],digits.target[train_index],digits.target[test_index]
    score_logistic.append((get_score(LogisticRegression(),X_train, X_test, y_train, y_test)))
    score_svm.append((get_score(SVC(),X_train, X_test, y_train, y_test)))
    score_rf.append((get_score(RandomForestClassifier(),X_train, X_test, y_train, y_test)))

print(score_logistic)
print(score_svm)
print(score_rf)

# print(cross_val_score(LogisticRegression(),digits.data,digits.target))
# print(cross_val_score(SVC(),digits.data,digits.target))
# print(cross_val_score(RandomForestClassifier(),digits.data,digits.target))






