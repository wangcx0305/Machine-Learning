#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 22:38:14 2016

@author: wangchunxiao
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

data = pd.read_table('proj_dat.txt', header = 0, sep = ',')
col_name = list(data.columns.values)




logrerate_001 = []
logrerate_005 = []
logrerate_01 = []
logrerate_05 = []
logrerate_1 = []
logrerate_5 = []
logrerate_10 = []
logrerate_25 = []
logrerate_50 = []
logrerate_100 = []

rfrate_5 = []
rfrate_10 = []
rfrate_25 = []
rfrate_50 = []
rfrate_100 = []


svmrate_lin = []
svmrate_rbf = []
svmrate_sig = []

nnrate_iden = []
nnrate_log = []
nnrate_tanh = []
nnrate_relu = []


Xlogrerate_001 = []
Xlogrerate_005 = []
Xlogrerate_01 = []
Xlogrerate_05 = []
Xlogrerate_1 = []
Xlogrerate_5 = []
Xlogrerate_10 = []
Xlogrerate_25 = []
Xlogrerate_50 = []
Xlogrerate_100 = []

Xrfrate_5 = []
Xrfrate_10 = []
Xrfrate_25 = []
Xrfrate_50 = []
Xrfrate_100 = []

Xsvmrate_lin = []
Xsvmrate_rbf = []
Xsvmrate_sig = []

Xnnrate_iden = []
Xnnrate_log = []
Xnnrate_tanh = []
Xnnrate_relu = []

Ylogrerate_001 = []
Ylogrerate_005 = []
Ylogrerate_01 = []
Ylogrerate_05 = []
Ylogrerate_1 = []
Ylogrerate_5 = []
Ylogrerate_10 = []
Ylogrerate_25 = []
Ylogrerate_50 = []
Ylogrerate_100 = []

Yrfrate_5 = []
Yrfrate_10 = []
Yrfrate_25 = []
Yrfrate_50 = []
Yrfrate_100 = []

Ysvmrate_lin = []
Ysvmrate_poly = []
Ysvmrate_rbf = []
Ysvmrate_sig = []

Ynnrate_iden = []
Ynnrate_log = []
Ynnrate_tanh = []
Ynnrate_relu = []

stlogrerate_001 = []
stlogrerate_005 = []
stlogrerate_01 = []
stlogrerate_05 = []
stlogrerate_1 = []
stlogrerate_5 = []
stlogrerate_10 = []
stlogrerate_25 = []
stlogrerate_50 = []
stlogrerate_100 = []

strfrate_5 = []
strfrate_10 = []
strfrate_25 = []
strfrate_50 = []
strfrate_100 = []

stsvmrate_lin = []
stsvmrate_rbf = []
stsvmrate_sig = []

stnnrate_iden = []
stnnrate_log = []
stnnrate_tanh = []
stnnrate_relu = []

for i in range(0, 10):
    
    train, test = train_test_split(data, test_size = 0.2)

    cretrain = np.array(train)
    cretest = np.array(test)


    trainy, trainX = cretrain[:,0], cretrain[:, 1:21]
    testy, testX = cretest[:,0], cretest[:, 1:21]
    trainy = np.array(trainy)
    testy = np.array(testy)
    
    
    #############logistic regression#############
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.01)
    logreclf.fit(trainX, trainy)
    logreprey = logreclf.predict(testX)
    logreinx = [i for i in range(0, 200) if logreprey[i] == testy[i]]
    logrerate_001.append(len(logreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.05)
    logreclf.fit(trainX, trainy)
    logreprey = logreclf.predict(testX)
    logreinx = [i for i in range(0, 200) if logreprey[i] == testy[i]]
    logrerate_005.append(len(logreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.1)
    logreclf.fit(trainX, trainy)
    logreprey = logreclf.predict(testX)
    logreinx = [i for i in range(0, 200) if logreprey[i] == testy[i]]
    logrerate_01.append(len(logreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.5)
    logreclf.fit(trainX, trainy)
    logreprey = logreclf.predict(testX)
    logreinx = [i for i in range(0, 200) if logreprey[i] == testy[i]]
    logrerate_05.append(len(logreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 1)
    logreclf.fit(trainX, trainy)
    logreprey = logreclf.predict(testX)
    logreinx = [i for i in range(0, 200) if logreprey[i] == testy[i]]
    logrerate_1.append(len(logreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 5)
    logreclf.fit(trainX, trainy)
    logreprey = logreclf.predict(testX)
    logreinx = [i for i in range(0, 200) if logreprey[i] == testy[i]]
    logrerate_5.append(len(logreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 10)
    logreclf.fit(trainX, trainy)
    logreprey = logreclf.predict(testX)
    logreinx = [i for i in range(0, 200) if logreprey[i] == testy[i]]
    logrerate_10.append(len(logreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 25)
    logreclf.fit(trainX, trainy)
    logreprey = logreclf.predict(testX)
    logreinx = [i for i in range(0, 200) if logreprey[i] == testy[i]]
    logrerate_25.append(len(logreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 50)
    logreclf.fit(trainX, trainy)
    logreprey = logreclf.predict(testX)
    logreinx = [i for i in range(0, 200) if logreprey[i] == testy[i]]
    logrerate_50.append(len(logreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 100)
    logreclf.fit(trainX, trainy)
    logreprey = logreclf.predict(testX)
    logreinx = [i for i in range(0, 200) if logreprey[i] == testy[i]]
    logrerate_100.append(len(logreinx) / len(cretest))
    
    
    
    ############random forest###################
    rfclf = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', max_features = 10)
    rfclf.fit(trainX, trainy)
    rfprey = rfclf.predict(testX)
    rfinx = [i for i in range(0, 200) if rfprey[i] == testy[i]]
    rfrate_5.append(len(rfinx) / len(cretest))
    
    rfclf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', max_features = 10)
    rfclf.fit(trainX, trainy)
    rfprey = rfclf.predict(testX)
    rfinx = [i for i in range(0, 200) if rfprey[i] == testy[i]]
    rfrate_10.append(len(rfinx) / len(cretest))
    
    rfclf = RandomForestClassifier(n_estimators = 25, criterion = 'entrophy', max_features = 10)
    rfclf.fit(trainX, trainy)
    rfprey = rfclf.predict(testX)
    rfinx = [i for i in range(0, 200) if rfprey[i] == testy[i]]
    rfrate_25.append(len(rfinx) / len(cretest))
    
    rfclf = RandomForestClassifier(n_estimators = 50, criterion = 'entrophy', max_features = 10)
    rfclf.fit(trainX, trainy)
    rfprey = rfclf.predict(testX)
    rfinx = [i for i in range(0, 200) if rfprey[i] == testy[i]]
    rfrate_50.append(len(rfinx) / len(cretest))
    
    rfclf = RandomForestClassifier(n_estimators = 100, criterion = 'entrophy', max_features = 10)
    rfclf.fit(trainX, trainy)
    rfprey = rfclf.predict(testX)
    rfinx = [i for i in range(0, 200) if rfprey[i] == testy[i]]
    rfrate_100.append(len(rfinx) / len(cretest))
    
    
    
    
    
    
    ###########support vector machine###########
    svmclf = svm.SVC(kernel = 'linear') 
    svmclf.fit(trainX, trainy)
    svmprey = svmclf.predict(testX)
    svminx = [i for i in range(0, 200) if svmprey[i] == testy[i]]
    svmrate_lin.append(len(svminx) / len(cretest))
    
    svmclf = svm.SVC(kernel = 'rbf')
    svmclf.fit(trainX, trainy)
    svmprey = svmclf.predict(testX)
    svminx = [i for i in range(0, 200) if svmprey[i] == testy[i]]
    svmrate_rbf.append(len(svminx) / len(cretest))
    
    svmclf = svm.SVC(kernel = 'sigmoid') 
    svmclf.fit(trainX, trainy)
    svmprey = svmclf.predict(testX)
    svminx = [i for i in range(0, 200) if svmprey[i] == testy[i]]
    svmrate_sig.append(len(svminx) / len(cretest))
    
    
    
    
    ###########neural network###############
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'identity',\
                          solver = 'sgd')
    nnclf.fit(trainX, trainy)
    nnprey = nnclf.predict(testX)
    nninx = [i for i in range(0, 200) if nnprey[i] == testy[i]]
    nnrate_iden.append(len(nninx) / len(cretest))
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'logistic',\
                          solver = 'sgd')
    nnclf.fit(trainX, trainy)
    nnprey = nnclf.predict(testX)
    nninx = [i for i in range(0, 200) if nnprey[i] == testy[i]]
    nnrate_log.append(len(nninx) / len(cretest))
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'tanh',\
                          solver = 'sgd')
    nnclf.fit(trainX, trainy)
    nnprey = nnclf.predict(testX)
    nninx = [i for i in range(0, 200) if nnprey[i] == testy[i]]
    nnrate_tanh.append(len(nninx) / len(cretest))
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'relu',\
                          solver = 'sgd')
    nnclf.fit(trainX, trainy)
    nnprey = nnclf.predict(testX)
    nninx = [i for i in range(0, 200) if nnprey[i] == testy[i]]
    nnrate_relu.append(len(nninx) / len(cretest))
    
    
    ##########extract important feature########
    clf = ExtraTreesClassifier()
    clf = clf.fit(trainX, trainy)
    importance = clf.feature_importances_
    #plt.plot(importance)
    #plt.xlabel('feature index')
    #plt.ylabel('importance score')
    model = SelectFromModel(clf, prefit = True)
    trainX_new = model.transform(trainX)
    nfeature = trainX_new.shape[1]
    testX_new = model.transform(testX)
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.01)
    logreclf.fit(trainX_new, trainy)
    Xlogreprey = logreclf.predict(testX_new)
    Xlogreinx = [i for i in range(0, 200) if Xlogreprey[i] == testy[i]]
    Xlogrerate_001.append(len(Xlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.05)
    logreclf.fit(trainX_new, trainy)
    Xlogreprey = logreclf.predict(testX_new)
    Xlogreinx = [i for i in range(0, 200) if Xlogreprey[i] == testy[i]]
    Xlogrerate_005.append(len(Xlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.1)
    logreclf.fit(trainX_new, trainy)
    Xlogreprey = logreclf.predict(testX_new)
    Xlogreinx = [i for i in range(0, 200) if Xlogreprey[i] == testy[i]]
    Xlogrerate_01.append(len(Xlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.5)
    logreclf.fit(trainX_new, trainy)
    Xlogreprey = logreclf.predict(testX_new)
    Xlogreinx = [i for i in range(0, 200) if Xlogreprey[i] == testy[i]]
    Xlogrerate_05.append(len(Xlogreinx) / len(cretest))
    
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 1)
    logreclf.fit(trainX_new, trainy)
    Xlogreprey = logreclf.predict(testX_new)
    Xlogreinx = [i for i in range(0, 200) if Xlogreprey[i] == testy[i]]
    Xlogrerate_1.append(len(Xlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 5)
    logreclf.fit(trainX_new, trainy)
    Xlogreprey = logreclf.predict(testX_new)
    Xlogreinx = [i for i in range(0, 200) if Xlogreprey[i] == testy[i]]
    Xlogrerate_5.append(len(Xlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 10)
    logreclf.fit(trainX_new, trainy)
    Xlogreprey = logreclf.predict(testX_new)
    Xlogreinx = [i for i in range(0, 200) if Xlogreprey[i] == testy[i]]
    Xlogrerate_10.append(len(Xlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 25)
    logreclf.fit(trainX_new, trainy)
    Xlogreprey = logreclf.predict(testX_new)
    Xlogreinx = [i for i in range(0, 200) if Xlogreprey[i] == testy[i]]
    Xlogrerate_25.append(len(Xlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 50)
    logreclf.fit(trainX_new, trainy)
    Xlogreprey = logreclf.predict(testX_new)
    Xlogreinx = [i for i in range(0, 200) if Xlogreprey[i] == testy[i]]
    Xlogrerate_50.append(len(Xlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 100)
    logreclf.fit(trainX_new, trainy)
    Xlogreprey = logreclf.predict(testX_new)
    Xlogreinx = [i for i in range(0, 200) if Xlogreprey[i] == testy[i]]
    Xlogrerate_100.append(len(Xlogreinx) / len(cretest))
    
    
    Xrfclf = RandomForestClassifier(n_estimators = 5, criterion = 'entrophy', max_features = int(nfeature / 2))
    Xrfclf.fit(trainX_new, trainy)
    Xrfprey = Xrfclf.predict(testX_new)
    Xrfinx = [i for i in range(0, 200) if Xrfprey[i] == testy[i]]
    Xrfrate_5.append(len(Xrfinx) / len(cretest))
    
    Xrfclf = RandomForestClassifier(n_estimators = 10, criterion = 'entrophy', max_features = int(nfeature / 2))
    Xrfclf.fit(trainX_new, trainy)
    Xrfprey = Xrfclf.predict(testX_new)
    Xrfinx = [i for i in range(0, 200) if Xrfprey[i] == testy[i]]
    Xrfrate_10.append(len(Xrfinx) / len(cretest))
    
    Xrfclf = RandomForestClassifier(n_estimators = 25, criterion = 'entrophy', max_features = int(nfeature / 2))
    Xrfclf.fit(trainX_new, trainy)
    Xrfprey = Xrfclf.predict(testX_new)
    Xrfinx = [i for i in range(0, 200) if Xrfprey[i] == testy[i]]
    Xrfrate_25.append(len(Xrfinx) / len(cretest))
    
    Xrfclf = RandomForestClassifier(n_estimators = 50, criterion = 'entrophy', max_features = int(nfeature / 2))
    Xrfclf.fit(trainX_new, trainy)
    Xrfprey = Xrfclf.predict(testX_new)
    Xrfinx = [i for i in range(0, 200) if Xrfprey[i] == testy[i]]
    Xrfrate_50.append(len(Xrfinx) / len(cretest))
    
    Xrfclf = RandomForestClassifier(n_estimators = 100, criterion = 'entrophy', max_features = int(nfeature / 2))
    Xrfclf.fit(trainX_new, trainy)
    Xrfprey = Xrfclf.predict(testX_new)
    Xrfinx = [i for i in range(0, 200) if Xrfprey[i] == testy[i]]
    Xrfrate_100.append(len(Xrfinx) / len(cretest))
    
    
    
    svmclf = svm.SVC(kernel = 'linear') 
    svmclf.fit(trainX_new, trainy)
    Xsvmprey = svmclf.predict(testX_new)
    Xsvminx = [i for i in range(0, 200) if Xsvmprey[i] == testy[i]]
    Xsvmrate_lin.append(len(Xsvminx) / len(cretest))
    
    svmclf = svm.SVC(kernel = 'rbf') 
    svmclf.fit(trainX_new, trainy)
    Xsvmprey = svmclf.predict(testX_new)
    Xsvminx = [i for i in range(0, 200) if Xsvmprey[i] == testy[i]]
    Xsvmrate_rbf.append(len(Xsvminx) / len(cretest))
    
    svmclf = svm.SVC(kernel = 'sigmoid') 
    svmclf.fit(trainX_new, trainy)
    Xsvmprey = svmclf.predict(testX_new)
    Xsvminx = [i for i in range(0, 200) if Xsvmprey[i] == testy[i]]
    Xsvmrate_sig.append(len(Xsvminx) / len(cretest))
    
    
    
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'identity',\
                          solver = 'sgd')
    nnclf.fit(trainX_new, trainy)
    Xnnprey = nnclf.predict(testX_new)
    Xnninx = [i for i in range(0, 200) if Xnnprey[i] == testy[i]]
    Xnnrate_iden.append(len(Xnninx) / len(cretest))
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'logistic',\
                          solver = 'sgd')
    nnclf.fit(trainX_new, trainy)
    Xnnprey = nnclf.predict(testX_new)
    Xnninx = [i for i in range(0, 200) if Xnnprey[i] == testy[i]]
    Xnnrate_log.append(len(Xnninx) / len(cretest))
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'tanh',\
                          solver = 'sgd')
    nnclf.fit(trainX_new, trainy)
    Xnnprey = nnclf.predict(testX_new)
    Xnninx = [i for i in range(0, 200) if Xnnprey[i] == testy[i]]
    Xnnrate_tanh.append(len(Xnninx) / len(cretest))
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'relu',\
                          solver = 'sgd')
    nnclf.fit(trainX_new, trainy)
    Xnnprey = nnclf.predict(testX_new)
    Xnninx = [i for i in range(0, 200) if Xnnprey[i] == testy[i]]
    Xnnrate_relu.append(len(Xnninx) / len(cretest))
    
    
    #########Tomek Link for labels###########
    
    tl = TomekLinks()
    trainX_tl, trainy_tl = tl.fit_sample(trainX, trainy)
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.01)
    logreclf.fit(trainX_tl, trainy_tl)
    Ylogreprey = logreclf.predict(testX)
    Ylogreinx = [i for i in range(0, 200) if Ylogreprey[i] == testy[i]]
    Ylogrerate_001.append(len(Ylogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.05)
    logreclf.fit(trainX_tl, trainy_tl)
    Ylogreprey = logreclf.predict(testX)
    Ylogreinx = [i for i in range(0, 200) if Ylogreprey[i] == testy[i]]
    Ylogrerate_005.append(len(Ylogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.1)
    logreclf.fit(trainX_tl, trainy_tl)
    Ylogreprey = logreclf.predict(testX)
    Ylogreinx = [i for i in range(0, 200) if Ylogreprey[i] == testy[i]]
    Ylogrerate_01.append(len(Ylogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.5)
    logreclf.fit(trainX_tl, trainy_tl)
    Ylogreprey = logreclf.predict(testX)
    Ylogreinx = [i for i in range(0, 200) if Ylogreprey[i] == testy[i]]
    Ylogrerate_05.append(len(Ylogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 1)
    logreclf.fit(trainX_tl, trainy_tl)
    Ylogreprey = logreclf.predict(testX)
    Ylogreinx = [i for i in range(0, 200) if Ylogreprey[i] == testy[i]]
    Ylogrerate_1.append(len(Ylogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 5)
    logreclf.fit(trainX_tl, trainy_tl)
    Ylogreprey = logreclf.predict(testX)
    Ylogreinx = [i for i in range(0, 200) if Ylogreprey[i] == testy[i]]
    Ylogrerate_5.append(len(Ylogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 10)
    logreclf.fit(trainX_tl, trainy_tl)
    Ylogreprey = logreclf.predict(testX)
    Ylogreinx = [i for i in range(0, 200) if Ylogreprey[i] == testy[i]]
    Ylogrerate_10.append(len(Ylogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 25)
    logreclf.fit(trainX_tl, trainy_tl)
    Ylogreprey = logreclf.predict(testX)
    Ylogreinx = [i for i in range(0, 200) if Ylogreprey[i] == testy[i]]
    Ylogrerate_25.append(len(Ylogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 50)
    logreclf.fit(trainX_tl, trainy_tl)
    Ylogreprey = logreclf.predict(testX)
    Ylogreinx = [i for i in range(0, 200) if Ylogreprey[i] == testy[i]]
    Ylogrerate_50.append(len(Ylogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 100)
    logreclf.fit(trainX_tl, trainy_tl)
    Ylogreprey = logreclf.predict(testX)
    Ylogreinx = [i for i in range(0, 200) if Ylogreprey[i] == testy[i]]
    Ylogrerate_100.append(len(Ylogreinx) / len(cretest))
    
    
    Yrfclf = RandomForestClassifier(n_estimators = 5, criterion = 'entrophy', max_features = 10)
    Yrfclf.fit(trainX_tl, trainy_tl)
    Yrfprey = Yrfclf.predict(testX)
    Yrfinx = [i for i in range(0, 200) if Yrfprey[i] == testy[i]]
    Yrfrate_5.append(len(Yrfinx) / len(cretest))
    
    Yrfclf = RandomForestClassifier(n_estimators = 10, criterion = 'entrophy', max_features = 10)
    Yrfclf.fit(trainX_tl, trainy_tl)
    Yrfprey = Yrfclf.predict(testX)
    Yrfinx = [i for i in range(0, 200) if Yrfprey[i] == testy[i]]
    Yrfrate_10.append(len(Yrfinx) / len(cretest))
    
    Yrfclf = RandomForestClassifier(n_estimators = 25, criterion = 'entrophy', max_features = 10)
    Yrfclf.fit(trainX_tl, trainy_tl)
    Yrfprey = Yrfclf.predict(testX)
    Yrfinx = [i for i in range(0, 200) if Yrfprey[i] == testy[i]]
    Yrfrate_25.append(len(Yrfinx) / len(cretest))
    
    Yrfclf = RandomForestClassifier(n_estimators = 50, criterion = 'entrophy', max_features = 10)
    Yrfclf.fit(trainX_tl, trainy_tl)
    Yrfprey = Yrfclf.predict(testX)
    Yrfinx = [i for i in range(0, 200) if Yrfprey[i] == testy[i]]
    Yrfrate_50.append(len(Yrfinx) / len(cretest))
    
    Yrfclf = RandomForestClassifier(n_estimators = 100, criterion = 'entrophy', max_features = 10)
    Yrfclf.fit(trainX_tl, trainy_tl)
    Yrfprey = Yrfclf.predict(testX)
    Yrfinx = [i for i in range(0, 200) if Yrfprey[i] == testy[i]]
    Yrfrate_100.append(len(Yrfinx) / len(cretest))
    
    
    
    svmclf = svm.SVC(kernel = 'linear') 
    svmclf.fit(trainX_tl, trainy_tl)
    Ysvmprey = svmclf.predict(testX)
    Ysvminx = [i for i in range(0, 200) if Ysvmprey[i] == testy[i]]
    Ysvmrate_lin.append(len(Ysvminx) / len(cretest))
    
    svmclf = svm.SVC(kernel = 'rbf') 
    svmclf.fit(trainX_tl, trainy_tl)
    Ysvmprey = svmclf.predict(testX)
    Ysvminx = [i for i in range(0, 200) if Ysvmprey[i] == testy[i]]
    Ysvmrate_rbf.append(len(Ysvminx) / len(cretest))
    
    svmclf = svm.SVC(kernel = 'sigmoid') 
    svmclf.fit(trainX_tl, trainy_tl)
    Ysvmprey = svmclf.predict(testX)
    Ysvminx = [i for i in range(0, 200) if Ysvmprey[i] == testy[i]]
    Ysvmrate_sig.append(len(Ysvminx) / len(cretest))
    
    
    
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'identity',\
                          solver = 'sgd')
    nnclf.fit(trainX_tl, trainy_tl)
    Ynnprey = nnclf.predict(testX)
    Ynninx = [i for i in range(0, 200) if Ynnprey[i] == testy[i]]
    Ynnrate_iden.append(len(Ynninx) / len(cretest))
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'logistic',\
                          solver = 'sgd')
    nnclf.fit(trainX_tl, trainy_tl)
    Ynnprey = nnclf.predict(testX)
    Ynninx = [i for i in range(0, 200) if Ynnprey[i] == testy[i]]
    Ynnrate_log.append(len(Ynninx) / len(cretest))
    
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'tanh',\
                          solver = 'sgd')
    nnclf.fit(trainX_tl, trainy_tl)
    Ynnprey = nnclf.predict(testX)
    Ynninx = [i for i in range(0, 200) if Ynnprey[i] == testy[i]]
    Ynnrate_tanh.append(len(Ynninx) / len(cretest))
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'relu',\
                          solver = 'sgd')
    nnclf.fit(trainX_tl, trainy_tl)
    Ynnprey = nnclf.predict(testX)
    Ynninx = [i for i in range(0, 200) if Ynnprey[i] == testy[i]]
    Ynnrate_relu.append(len(Ynninx) / len(cretest))
    
    
    
    ###########SMOTE + TL for imbalanced data###########
    st = SMOTETomek()
    trainX_st, trainy_st = st.fit_sample(trainX, trainy)
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.01)
    logreclf.fit(trainX_st, trainy_st)
    stlogreprey = logreclf.predict(testX)
    stlogreinx = [i for i in range(0, 200) if stlogreprey[i] == testy[i]]
    stlogrerate_001.append(len(stlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.05)
    logreclf.fit(trainX_st, trainy_st)
    stlogreprey = logreclf.predict(testX)
    stlogreinx = [i for i in range(0, 200) if stlogreprey[i] == testy[i]]
    stlogrerate_005.append(len(stlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.1)
    logreclf.fit(trainX_st, trainy_st)
    stlogreprey = logreclf.predict(testX)
    stlogreinx = [i for i in range(0, 200) if stlogreprey[i] == testy[i]]
    stlogrerate_01.append(len(stlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 0.5)
    logreclf.fit(trainX_st, trainy_st)
    stlogreprey = logreclf.predict(testX)
    stlogreinx = [i for i in range(0, 200) if stlogreprey[i] == testy[i]]
    stlogrerate_05.append(len(stlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 1)
    logreclf.fit(trainX_st, trainy_st)
    stlogreprey = logreclf.predict(testX)
    stlogreinx = [i for i in range(0, 200) if stlogreprey[i] == testy[i]]
    stlogrerate_1.append(len(stlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 5)
    logreclf.fit(trainX_st, trainy_st)
    stlogreprey = logreclf.predict(testX)
    stlogreinx = [i for i in range(0, 200) if stlogreprey[i] == testy[i]]
    stlogrerate_5.append(len(stlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 10)
    logreclf.fit(trainX_st, trainy_st)
    stlogreprey = logreclf.predict(testX)
    stlogreinx = [i for i in range(0, 200) if stlogreprey[i] == testy[i]]
    stlogrerate_10.append(len(stlogreinx) / len(cretest))
    
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 25)
    logreclf.fit(trainX_st, trainy_st)
    stlogreprey = logreclf.predict(testX)
    stlogreinx = [i for i in range(0, 200) if stlogreprey[i] == testy[i]]
    stlogrerate_25.append(len(stlogreinx) / len(cretest))
    
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 50)
    logreclf.fit(trainX_st, trainy_st)
    stlogreprey = logreclf.predict(testX)
    stlogreinx = [i for i in range(0, 200) if stlogreprey[i] == testy[i]]
    stlogrerate_50.append(len(stlogreinx) / len(cretest))
    
    logreclf = linear_model.LogisticRegression(penalty = 'l1', C = 100)
    logreclf.fit(trainX_st, trainy_st)
    stlogreprey = logreclf.predict(testX)
    stlogreinx = [i for i in range(0, 200) if stlogreprey[i] == testy[i]]
    stlogrerate_100.append(len(stlogreinx) / len(cretest))
    
    
    
    
    
    strfclf = RandomForestClassifier(n_estimators = 5, criterion = 'entrophy', max_features = 10)
    strfclf.fit(trainX_st, trainy_st)
    strfprey = strfclf.predict(testX)
    strfinx = [i for i in range(0, 200) if strfprey[i] == testy[i]]
    strfrate_5.append(len(strfinx) / len(cretest))
    
    strfclf = RandomForestClassifier(n_estimators = 10, criterion = 'entrophy', max_features = 10)
    strfclf.fit(trainX_st, trainy_st)
    strfprey = strfclf.predict(testX)
    strfinx = [i for i in range(0, 200) if strfprey[i] == testy[i]]
    strfrate_10.append(len(strfinx) / len(cretest))
    
    strfclf = RandomForestClassifier(n_estimators = 25, criterion = 'entrophy', max_features = 10)
    strfclf.fit(trainX_st, trainy_st)
    strfprey = strfclf.predict(testX)
    strfinx = [i for i in range(0, 200) if strfprey[i] == testy[i]]
    strfrate_25.append(len(strfinx) / len(cretest))
    
    strfclf = RandomForestClassifier(n_estimators = 50, criterion = 'entrophy', max_features = 10)
    strfclf.fit(trainX_st, trainy_st)
    strfprey = strfclf.predict(testX)
    strfinx = [i for i in range(0, 200) if strfprey[i] == testy[i]]
    strfrate_50.append(len(strfinx) / len(cretest))
    
    strfclf = RandomForestClassifier(n_estimators = 100, criterion = 'entrophy', max_features = 10)
    strfclf.fit(trainX_st, trainy_st)
    strfprey = strfclf.predict(testX)
    strfinx = [i for i in range(0, 200) if strfprey[i] == testy[i]]
    strfrate_100.append(len(strfinx) / len(cretest))
    
    svmclf = svm.SVC(kernel = 'linear')
    svmclf.fit(trainX_st, trainy_st)
    stsvmprey = svmclf.predict(testX)
    stsvminx = [i for i in range(0, 200) if stsvmprey[i] == testy[i]]
    stsvmrate_lin.append(len(stsvminx) / len(cretest))
    
    svmclf = svm.SVC(kernel = 'rbf')
    svmclf.fit(trainX_st, trainy_st)
    stsvmprey = svmclf.predict(testX)
    stsvminx = [i for i in range(0, 200) if stsvmprey[i] == testy[i]]
    stsvmrate_rbf.append(len(stsvminx) / len(cretest))
    
    svmclf = svm.SVC(kernel = 'sigmoid')
    svmclf.fit(trainX_st, trainy_st)
    stsvmprey = svmclf.predict(testX)
    stsvminx = [i for i in range(0, 200) if stsvmprey[i] == testy[i]]
    stsvmrate_sig.append(len(stsvminx) / len(cretest))
    
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'identity',\
                          solver = 'sgd')
    nnclf.fit(trainX_st, trainy_st)
    stnnprey = nnclf.predict(testX)
    stnninx = [i for i in range(0, 200) if stnnprey[i] == testy[i]]
    stnnrate_iden.append(len(stnninx) / len(cretest))
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'logistic',\
                          solver = 'sgd')
    nnclf.fit(trainX_st, trainy_st)
    stnnprey = nnclf.predict(testX)
    stnninx = [i for i in range(0, 200) if stnnprey[i] == testy[i]]
    stnnrate_log.append(len(stnninx) / len(cretest))
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'tanh',\
                          solver = 'sgd')
    nnclf.fit(trainX_st, trainy_st)
    stnnprey = nnclf.predict(testX)
    stnninx = [i for i in range(0, 200) if stnnprey[i] == testy[i]]
    stnnrate_tanh.append(len(stnninx) / len(cretest))
    
    nnclf = MLPClassifier(hidden_layer_sizes = (3, 3), activation = 'relu',\
                          solver = 'sgd')
    nnclf.fit(trainX_st, trainy_st)
    stnnprey = nnclf.predict(testX)
    stnninx = [i for i in range(0, 200) if stnnprey[i] == testy[i]]
    stnnrate_relu.append(len(stnninx) / len(cretest))
    
    
avelogrerate = []   

avelogrerate.append(np.mean(logrerate_001))
avelogrerate.append(np.mean(logrerate_005))
avelogrerate.append(np.mean(logrerate_01))
avelogrerate.append(np.mean(logrerate_05))
avelogrerate.append(np.mean(logrerate_1))
avelogrerate.append(np.mean(logrerate_5))
avelogrerate.append(np.mean(logrerate_10))
avelogrerate.append(np.mean(logrerate_25))
avelogrerate.append(np.mean(logrerate_50))
avelogrerate.append(np.mean(logrerate_100))


aveXlogrerate = []

aveXlogrerate.append(np.mean(Xlogrerate_001))
aveXlogrerate.append(np.mean(Xlogrerate_005))
aveXlogrerate.append(np.mean(Xlogrerate_01))
aveXlogrerate.append(np.mean(Xlogrerate_05))
aveXlogrerate.append(np.mean(Xlogrerate_1))
aveXlogrerate.append(np.mean(Xlogrerate_5))
aveXlogrerate.append(np.mean(Xlogrerate_10))
aveXlogrerate.append(np.mean(Xlogrerate_25))
aveXlogrerate.append(np.mean(Xlogrerate_50))
aveXlogrerate.append(np.mean(Xlogrerate_100))

aveYlogrerate = []

aveYlogrerate.append(np.mean(Ylogrerate_001))
aveYlogrerate.append(np.mean(Ylogrerate_005))
aveYlogrerate.append(np.mean(Ylogrerate_01))
aveYlogrerate.append(np.mean(Ylogrerate_05))
aveYlogrerate.append(np.mean(Ylogrerate_1))
aveYlogrerate.append(np.mean(Ylogrerate_5))
aveYlogrerate.append(np.mean(Ylogrerate_10))
aveYlogrerate.append(np.mean(Ylogrerate_25))
aveYlogrerate.append(np.mean(Ylogrerate_50))
aveYlogrerate.append(np.mean(Ylogrerate_100))


avestlogrerate = []

avestlogrerate.append(np.mean(stlogrerate_001))
avestlogrerate.append(np.mean(stlogrerate_005))
avestlogrerate.append(np.mean(stlogrerate_01))
avestlogrerate.append(np.mean(stlogrerate_05))
avestlogrerate.append(np.mean(stlogrerate_1))
avestlogrerate.append(np.mean(stlogrerate_5))
avestlogrerate.append(np.mean(stlogrerate_10))
avestlogrerate.append(np.mean(stlogrerate_25))
avestlogrerate.append(np.mean(stlogrerate_50))
avestlogrerate.append(np.mean(stlogrerate_100))


averfrate = []

averfrate.append(np.mean(rfrate_5))
averfrate.append(np.mean(rfrate_10))
averfrate.append(np.mean(rfrate_25))
averfrate.append(np.mean(rfrate_50))
averfrate.append(np.mean(rfrate_100))

aveXrfrate = []

aveXrfrate.append(np.mean(Xrfrate_5))
aveXrfrate.append(np.mean(Xrfrate_10))
aveXrfrate.append(np.mean(Xrfrate_25))
aveXrfrate.append(np.mean(Xrfrate_50))
aveXrfrate.append(np.mean(Xrfrate_100))

aveYrfrate = []

aveYrfrate.append(np.mean(Yrfrate_5))
aveYrfrate.append(np.mean(Yrfrate_10))
aveYrfrate.append(np.mean(Yrfrate_25))
aveYrfrate.append(np.mean(Yrfrate_50))
aveYrfrate.append(np.mean(Yrfrate_100))

avestrfrate = []

avestrfrate.append(np.mean(strfrate_5))
avestrfrate.append(np.mean(strfrate_10))
avestrfrate.append(np.mean(strfrate_25))
avestrfrate.append(np.mean(strfrate_50))
avestrfrate.append(np.mean(strfrate_100))

avesvmrate = []

avesvmrate.append(np.mean(svmrate_lin))
avesvmrate.append(np.mean(svmrate_rbf))
avesvmrate.append(np.mean(svmrate_sig))

aveXsvmrate = []

aveXsvmrate.append(np.mean(Xsvmrate_lin))
aveXsvmrate.append(np.mean(Xsvmrate_rbf))
aveXsvmrate.append(np.mean(Xsvmrate_sig))

aveYsvmrate = []

aveYsvmrate.append(np.mean(Ysvmrate_lin))
aveYsvmrate.append(np.mean(Ysvmrate_rbf))
aveYsvmrate.append(np.mean(Ysvmrate_sig))

avestsvmrate = []

avestsvmrate.append(np.mean(stsvmrate_lin))
avestsvmrate.append(np.mean(stsvmrate_rbf))
avestsvmrate.append(np.mean(stsvmrate_sig))

avennrate = []

avennrate.append(np.mean(nnrate_iden))
avennrate.append(np.mean(nnrate_log))
avennrate.append(np.mean(nnrate_tanh))
avennrate.append( np.mean(nnrate_relu))

aveXnnrate = []

aveXnnrate.append(np.mean(Xnnrate_iden))
aveXnnrate.append(np.mean(Xnnrate_log))
aveXnnrate.append(np.mean(Xnnrate_tanh))
aveXnnrate.append( np.mean(Xnnrate_relu))

aveYnnrate = []

aveYnnrate.append(np.mean(Ynnrate_iden))
aveYnnrate.append(np.mean(Ynnrate_log))
aveYnnrate.append(np.mean(Ynnrate_tanh))
aveYnnrate.append( np.mean(Ynnrate_relu))


avestnnrate = []

avestnnrate.append(np.mean(stnnrate_iden))
avestnnrate.append(np.mean(stnnrate_log))
avestnnrate.append(np.mean(stnnrate_tanh))
avestnnrate.append( np.mean(stnnrate_relu))

lambda_logistic = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 100]
lambda_logistic_df = pd.DataFrame(lambda_logistic)
lambda_logistic_df.to_csv('lambda_logistic_csv', index = False, header = False)

logrerate_df = pd.DataFrame(avelogrerate)
logrerate_df.to_csv('logrerate_csv', index = False, header = False)

featurelogrerate_df = pd.DataFrame(aveXlogrerate)
featurelogrerate_df.to_csv('featurelogrerate_csv', index = False, header = False)

labellogrerate_df = pd.DataFrame(aveYlogrerate)
labellogrerate_df.to_csv('labellogrerate_csv', index = False, header = False)

smotelogrerate_df = pd.DataFrame(avestlogrerate)
smotelogrerate_df.to_csv('smotelogrerate_csv', index = False, header = False)

random_forest_rate_df = pd.DataFrame(averfrate)
random_forest_rate_df.to_csv('random_forest_rate_csv', index = False, header = False)

feature_random_forest_rate_df = pd.DataFrame(aveXrfrate)
feature_random_forest_rate_df.to_csv('feature_random_forest_rate_csv', index = False, header = False)

label_random_forest_rate_df = pd.DataFrame(aveYrfrate)
label_random_forest_rate_df.to_csv('label_random_forest_rate_csv', index = False, header = False)

smote_random_forest_rate_df = pd.DataFrame(avestrfrate)
smote_random_forest_rate_df.to_csv('smote_random_forest_rate_csv', index = False, header = False)

svm_rate_df = pd.DataFrame(avesvmrate)
svm_rate_df.to_csv('svm_rate_csv', index = False, header = False)

feature_svm_rate_df = pd.DataFrame(aveXsvmrate)
feature_svm_rate_df.to_csv('feature_svm_rate_csv', index = False, header = False)

label_svm_rate_df = pd.DataFrame(aveYsvmrate)
label_svm_rate_df.to_csv('label_svm_rate_csv', index = False, header = False)

smote_svm_rate_df = pd.DataFrame(avestsvmrate)
smote_svm_rate_df.to_csv('smote_svm_rate_csv', index = False, header = False)

neural_rate_df = pd.DataFrame(avennrate)
neural_rate_df.to_csv('neural_rate_csv', index = False, header = False)

feature_neural_rate_df = pd.DataFrame(aveXnnrate)
feature_neural_rate_df.to_csv('feature_neural_rate_csv', index = False, header = False)

label_neural_rate_df = pd.DataFrame(aveYnnrate)
label_neural_rate_df.to_csv('label_neural_rate_csv', index = False, header = False)

smote_neural_rate_df = pd.DataFrame(avestnnrate)
smote_neural_rate_df.to_csv('smote_neural_rate_csv', index = False, header = False)
