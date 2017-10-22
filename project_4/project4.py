#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:50:18 2016

@author: wangchunxiao
"""

import pandas as pd
import numpy as np
import random

#######################problem 1#########################################
#######################problem 1#########################################

train = np.array(pd.read_table('walkingdata', header = None, sep = ' '))
label = np.array(pd.read_table('walkinglabels', header = None, sep = ' '))

def assignclusters(data, mu):
    clusters  = {}
    for elem in data:
        d = [np.linalg.norm(elem - m) for m in mu]
        ca = d.index(min(d))
        try:
            clusters[ca].append(elem)
        except KeyError:
            clusters[ca] = [elem]
    return clusters
 
def recenters(mu, clusters):
    newmu = list();
    keys = sorted(clusters.keys())
    for ca in keys:
        newmu.append(np.mean(clusters[ca], axis = 0))
    return newmu
 
def stop(mu, oldmu):
    if (type(mu) == type(oldmu) == 'list'):
        return mu == oldmu
    elif (type(oldmu) == 'list'):
        temp = oldmu.tolist()
        return temp == mu
    else:
        return np.array_equal(mu, oldmu)
            
 
def findcluster(data, k):
    oldmu = data[np.random.choice(data.shape[0], k)]
    mu = data[np.random.choice(data.shape[0], k)]
    while not stop(mu, oldmu):
        oldmu = mu
        clusters = assignclusters(data, mu)
        mu = recenters(oldmu, clusters)
    return(mu, clusters)
    
random.seed(1123)
mulist = []
purlist = []
for i in range(0, 10):   
    cluster = findcluster(train, 2)[1] 
    mu = findcluster(train, 2)[0]
    cluster1 = cluster[0]
    cluster2 = cluster[1]
    cluster1label = [];
    cluster2label = [];
    
    for i in range(0, len(cluster1)):
        for j in range(0, train.shape[0]):
            if np.array_equal(cluster1[i], train[j]):
                cluster1label.append(label[j])
                
    cluster2label = np.delete(label, cluster1label)
    
    pur1 = max(sum(cluster1label), len(cluster1label) - sum(cluster1label))
    pur2 = max(sum(cluster2label), len(cluster2label) - sum(cluster2label))
    purlist.append((pur1 + pur2) / len(train))
    mulist.append(mu)

purlist

########################problem 3##################################
###################################################################
xpca1 = np.array(pd.read_csv('X_pca1.csv', header = None, sep = ' '))
xpca2 = np.array(pd.read_csv('X_pca2.csv', header = None, sep = ','))
xpca3 = np.array(pd.read_csv('X_pca3.csv', header = None, sep = ','))

random.seed(1123)
purlist1 = []
purlist2 = []
purlist3 = []

for i in range(0, 10):   
    cluster = findcluster(xpca1, 2)[1] 
    mu = findcluster(xpca1, 2)[0]
    cluster1 = cluster[0]
    cluster2 = cluster[1]
    cluster1label = [];
    cluster2label = [];
    
    for i in range(0, len(cluster1)):
        for j in range(0, xpca1.shape[0]):
            if np.array_equal(cluster1[i], xpca1[j]):
                cluster1label.append(label[j])
                
    cluster2label = np.delete(label, cluster1label)
    
    pur1 = max(sum(cluster1label), len(cluster1label) - sum(cluster1label))
    pur2 = max(sum(cluster2label), len(cluster2label) - sum(cluster2label))
    purlist1.append((pur1 + pur2) / len(xpca1))
    

for i in range(0, 10):   
    cluster = findcluster(xpca2, 2)[1] 
    mu = findcluster(xpca2, 2)[0]
    cluster1 = cluster[0]
    cluster2 = cluster[1]
    cluster1label = [];
    cluster2label = [];
    
    for i in range(0, len(cluster1)):
        for j in range(0, xpca2.shape[0]):
            if np.array_equal(cluster1[i], xpca2[j]):
                cluster1label.append(label[j])
                
    cluster2label = np.delete(label, cluster1label)
    
    pur1 = max(sum(cluster1label), len(cluster1label) - sum(cluster1label))
    pur2 = max(sum(cluster2label), len(cluster2label) - sum(cluster2label))
    purlist2.append((pur1 + pur2) / len(xpca2))
    

for i in range(0, 10):   
    cluster = findcluster(xpca3, 2)[1] 
    mu = findcluster(xpca3, 2)[0]
    cluster1 = cluster[0]
    cluster2 = cluster[1]
    cluster1label = [];
    cluster2label = [];
    
    for i in range(0, len(cluster1)):
        for j in range(0, xpca3.shape[0]):
            if np.array_equal(cluster1[i], xpca3[j]):
                cluster1label.append(label[j])
                
    cluster2label = np.delete(label, cluster1label)
    
    pur1 = max(sum(cluster1label), len(cluster1label) - sum(cluster1label))
    pur2 = max(sum(cluster2label), len(cluster2label) - sum(cluster2label))
    purlist3.append((pur1 + pur2) / len(xpca3))


purlist1
purlist2
purlist3



######################Problem 4##########################################
#########################################################################
data = np.concatenate((train, label), axis = 1)
cluster0 = np.array([i for i in data if i[477] == 0])
cluster1 = np.array([i for i in data if i[477] == 1])
cluster0 = cluster0[:, 0:477]
cluster1 = cluster1[:, 0:477]

m0 = np.array(np.mean(cluster0, axis = 0))
m1 = np.array(np.mean(cluster1, axis = 0))

centercluster0 = np.matrix([i - m0 for i in cluster0])
centercluster1 = np.matrix([i - m1 for i in cluster1])

S = centercluster0.T * centercluster0 + centercluster1.T * centercluster1
mdiff = np.matrix(m0 - m1).T

smallterm =  10^-6
w = np.linalg.inv(S + np.eye(S.shape[1]) * smallterm) * mdiff
data = np.array(np.matrix(train) * w)
                             
random.seed(1123)
purlistlda = []

for i in range(0, 10):   
    cluster = findcluster(data, 2)[1] 
    mu = findcluster(data, 2)[0]
    cluster1 = cluster[0]
    cluster2 = cluster[1]
    cluster1label = [];
    cluster2label = [];
    
    for i in range(0, len(cluster1)):
        for j in range(0, data.shape[0]):
            if np.array_equal(cluster1[i], data[j]):
                cluster1label.append(label[j])
                
    cluster2label = np.delete(label, cluster1label)
    
    pur1 = max(sum(cluster1label), len(cluster1label) - sum(cluster1label))
    pur2 = max(sum(cluster2label), len(cluster2label) - sum(cluster2label))
    purlistlda.append((pur1 + pur2) / len(data))



  
    
    
    
    