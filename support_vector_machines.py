# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 2017

@author: Amrut Rajkarne
"""

# Classification from: http://www.scipy-lectures.org/advanced/scikit-learn/#id2

# Importing Libraries
import pandas as pd 
import numpy as np
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

"""
# STEP 1: DATA PREPROCESSING
"""
# Add any file name
df = pd.read_csv(open('sample_data.csv'))
df=df.dropna()

#  Add dummy variables for categorical variables
df_dist_dummies = pd.get_dummies(df['distribution'])
df = pd.concat([df, df_dist_dummies], axis=1)

# Change target variable to int
df.good =df.good.astype(int)
y=df.good
y=y.ravel()

# Define your predictors
predictors =  ['min','max','mean','std','q25', 'q50', 'q75', 'length', 's_min', 's_max', 's_mean', 's_std', 's_q25', 's_q50', 's_q75', 't_min', 't_max', 't_mean', 't_std', 't_q25', 't_q50', 't_q75', 'beta', 'chi2', 'exp', 'gamma', 'geometric', 'laplace', 'logistic', 'lognormal', 'negbin', 'normal', 'poisson', 'student', 'uniform', 'weibull']

# Randomly split dataset to training and testing samples
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df.good, test_size=0.2, random_state=0)


# FUTURE STEPS: CHECK FOR MULTICOLLINEARITY + IMPLEMENT FEATURE SELECTION
"""
# STEP 2: GRID SEARCH TO SELECT BEST KERNEL AND PARAMETERS
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

"""

tuned_parameters = [
  {'C': [1,10,100,1000], 'kernel': ['linear']},
  {'C': [1,10,100,1000], 'gamma': [0.01, 0.001, 0.0001,0.00001], 'kernel': ['rbf']},
 ]

clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=None)
clf.fit(X_train, y_train.ravel())
#clf.fit(df[predictors], df.good)
y_true, y_pred = y_test, clf.predict(X_test)

#confusion matrix
print(confusion_matrix(y_true, y_pred))
#accuracy score
print(accuracy_score(y_true,  y_pred))

#Best Parameters for SVM
print("Best parameters set found on development set:")
print(clf.best_params_)

# Adaboost : Boosting SVM with the best paramters
# Boosting generally doesn't work very well with strong classifiers like SVM
clf_boost = AdaBoostClassifier(svm.SVC(kernel='rbf', C=1000, gamma=1e-05 ,probability=True), n_estimators=50, learning_rate=1.0, algorithm='SAMME')
clf_boost.fit(X_train, y_train.ravel())
y_true, y_boost_pred = y_test, clf_boost.predict(X_test)

#confusion matrix for Adaboost
print(confusion_matrix(y_true, y_boost_pred))
#accuracy score for Adaboost
print(accuracy_score(y_true,  y_boost_pred))
