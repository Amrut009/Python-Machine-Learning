# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 2017
@author: Amrut Rajkarne
"""

# Importing Libraries
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

"""
# STEP 1: DATA PREPROCESSING
"""

# Train data - Add any file name
df_train = pd.read_csv(open('data.csv'))
df_train = df_train.dropna()

# Test data - Add any file name
df_test = pd.read_csv(open('test.csv'))
df_test = df_test.dropna()

# If test data is not available separately then use train-test split from scikit-learn

# Change target variable to int
df_train.target = df_train.target.astype(int)
Y_train = df_train.target
Y_train = Y_train.ravel()

df_test.target = df_test.target.astype(int)
Y_test = df_test.target
Y_test = Y_test.ravel()

# Define your predictors
predictors =  ['feature1','feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21']

# Training and Testing samples
X_train = df_train[predictors]
X_test = df_test[predictors]


"""
# STEP 2: LOGISTIC REGRESSION
"""

model = LogisticRegression()
model = model.fit(X_train, Y_train)
print "Accuracy for training data using Logisitc Regression - ", model.score(X_train, Y_train)
print ""

# Generate class probabilities
probs = model.predict_proba(X_test)

# Predict class labels for the test set
predicted = model.predict(X_test)


"""
# STEP 3: PERFORMANCE METRICS
"""

# Generate Log loss for the model
print "Logloss for test data using Logisitic Regression - ", metrics.log_loss(Y_test, probs)
print ""

# Other performace metrics
print "Accuracy for test data using Logisitic Regression - ", metrics.accuracy_score(Y_test, predicted)
print ""
print "AUC ROC for test data using Logisitic Regression -", metrics.roc_auc_score(Y_test, probs[:, 1])
print ""

print "Confusion matrix for test data using Logisitic Regression"
print metrics.confusion_matrix(Y_test, predicted)
print ""
print "Classification Report for test data using Logistic Regression"
print metrics.classification_report(Y_test, predicted)

"""
# STEP 4: EXPORT TEST RESULTS
"""
df_output = df_test[["ident","target"]].copy()
df_output['Probability0'] = probs[:,0]
df_output['Probability1'] = probs[:,1]
df_output.to_csv("results.csv")



