#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:47:49 2016

@author: wangchunxiao
"""

import pandas as pd
import numpy as np
from math import log
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

train = np.array(pd.read_csv('iris_train.csv', header = None, sep = ';'))
test = np.array(pd.read_csv('iris_test.csv', header = None, sep = ';'))


###############Problem 1###############################################

def dividedata(data, feature, value):
    set1 = [element for element in data if element[feature] < value ]
    set2 = [element for element in data if element[feature] >= value]
    return (set1, set2)

def categorynum(data):   
    category = {}
    for element in data:
        ca = element[len(element) - 1]
        if ca not in category:
            category[ca] = 1
        else:
            category[ca] = category[ca] + 1
    return category
    
def entrophy(data):
    calist = categorynum(data)
    ent = 0
    for cat in calist.keys():
        p = float(calist[cat] / len(data))
        if p == 0:
            ent = ent
        else:
            ent = ent - p * log(p) / log(2)
    return ent
    
class node:
    def __init__(self, feature = None, value = None, gain = None, result = None, lchild = None, rchild = None):
        self.feature = feature
        self.value = value
        self.gain = gain
        self.result = result
        self.lchild = lchild
        self.rchild = rchild
            

def dt(data, k):
    if len(data) < k: 
        return    
    curentro = entrophy(data)
    best_gain = 0
    best_criteria = None
    best_set = None
    for i in range(0, len(data[0]) - 1):
        values = {}
        for element in data:
            values[element[i]] = 1
        for j in values.keys():
            (set1, set2) = dividedata(data, i, j)
            p = float(len(set1) / len(data))
            if p == 0 or p == 1:
                gain = 0
            else:
                gain = curentro - p * entrophy(set1) - (1 - p) * entrophy(set2)
            if gain > best_gain:
                best_gain = gain
                best_set = (set1, set2)
                best_criteria = (i, j)                         
    if best_gain > 0 and len(best_set[0]) >= k and len(best_set[1]) >= k:
         ltree = dt(best_set[0], k)
         rtree = dt(best_set[1], k)
         return node(feature = best_criteria[0], value = best_criteria[1], gain = round(best_gain, 3), lchild = ltree, rchild = rtree)
    else:
         return node(result = categorynum(data))
        
                  
def printtree(tree, indent = " "):
     if tree.result != None:
          print(str(tree.result))
     else:
          print(str(tree.feature) + " < " + str(tree.value) + "?" + "," + "gain = " + str(tree.gain))
          print(indent + "Left->", end = " ")
          printtree(tree.lchild, indent + " ")
          print(indent + "Right->", end = " ")
          printtree(tree.rchild, indent + " ")
          
          
def getwidth(tree):
  if tree.lchild == None and tree.rchild == None: return 1
  return getwidth(tree.lchild) + getwidth(tree.rchild)

def getdepth(tree):
  if tree.lchild == None and tree.rchild == None: return 0
  return max(getdepth(tree.lchild), getdepth(tree.rchild)) + 1


def drawtree(tree, jpeg='tree.jpg'):
  w = getwidth(tree) * 100
  h = getdepth(tree) * 100 + 120

  img = Image.new('RGB', (w,h), (255,255,255))
  draw = ImageDraw.Draw(img)

  drawnode(draw, tree, w/2, 20)
  img.save(jpeg, 'JPEG')
  
def drawnode(draw,tree,x,y):
  if tree.result == None:
    w1 = getwidth(tree.lchild) * 100
    w2 = getwidth(tree.rchild) * 100

    left = x - (w1 + w2) / 2
    right = x + (w1 + w2) / 2

    draw.text((x - 20,y - 10), str(tree.feature) + '<' + str(tree.value) + ',' +'gain = ' + str(tree.gain), (0,0,0))

    draw.line((x, y, left + w1 / 2, y + 100), fill = (255,0,0))
    draw.line((x, y, right - w2 / 2,y + 100), fill = (255,0,0))

    drawnode(draw,tree.lchild, left + w1 / 2, y + 100)
    drawnode(draw,tree.rchild, right - w2/2, y + 100)
  else:
    txt=' \n'.join(['%s : %d'%v for v in tree.result.items()])
    draw.text((x - 20, y), txt, (0, 0, 0))

def predict(tree, data):
      if tree.result != None:
         return max(tree.result, key = tree.result.get)
      else:
        subtree = None
        if data[tree.feature] < tree.value:
            subtree = tree.lchild
        else:
            subtree = tree.rchild    
        return predict(subtree, data)
        
def predictdataset(tree, data):
    result = list() ;
    for i in range(0, len(data)):
        cat = predict(tree, data[i])
        result.append(cat)
    return result
    
traintrueresult = train[:, 4]
testtrueresult = test[:, 4]
trainerrate = list()
testerrate = list()
for k in range(1, 11):
    tree = dt(train, k)
    trainresult = predictdataset(tree, train)
    testresult = predictdataset(tree, test)
    trainerrate.append(sum(traintrueresult != trainresult) / len(train))
    testerrate.append(sum(testtrueresult != testresult) / len(test))

x = range(1, 11)
plt.plot(x, trainerrate)
plt.plot(x, testerrate)
plt.xlabel('k')
plt.ylabel('error rate')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



###############Problem 2#####################################################
###############Also use dividedata, entrophy, predict, predictdataset in 1###  
    
def feabaggingt(data, k):
    if len(data) < k: 
        return
    else:
        curentro = entrophy(data)
        best_gain = 0
        best_criteria = None
        best_set = None
        ranfea = np.random.choice(range(0, len(test[0]) - 1), 2, replace = False)
        for i in ranfea:
            values = {}
            for element in data:
                values[element[i]] = 1
            for j in values.keys():
                (set1, set2) = dividedata(data, i, j)
                p = float(len(set1) / len(data))
                if p == 0 or p == 1:
                    gain = 0
                else:
                  gain = curentro - p * entrophy(set1) - (1 - p) * entrophy(set2)
                if gain > best_gain:
                    best_gain = gain
                    best_set = (set1, set2)
                    best_criteria = (i, j)
        if best_gain > 0 and len(best_set[0]) >= k and len(best_set[1]) >= k:
             ltree = feabaggingt(best_set[0], k)
             rtree = feabaggingt(best_set[1], k)
             return node(feature = best_criteria[0], value = best_criteria[1], gain = best_gain, lchild = ltree, rchild = rtree)
        else:
             return node(result = categorynum(data))
                
      
def rfclassification(L, data, k, predata):
    result = list()
    pred = np.empty((len(predata), L))
    pred[:] = np.nan
    for i in range(0, L):
        a = np.random.choice(range(0, len(data)), len(data), replace = True)
        selectdata = [data[i] for i in a]
        tree = feabaggingt(selectdata, k)
        pred[:, i] = predictdataset(tree, predata)
    for j in range(0, len(pred)):
        lst = pred[j].tolist()
        result.append(max(set(lst), key = lst.count))
    return result
        
def averageerrate(L, data, k, predata):
       trueresult = predata[:, 4] 
       errate = 0 
       for i in range(0, 10):
           preresult = rfclassification(L, data, k, predata)
           errate = errate + sum(trueresult != preresult) / len(predata)
       return errate / 10
       
           
trainarate5 = [averageerrate(5, train, k, train) for k in range(1, 11)]
testarate5 =  [averageerrate(5, train, k, test) for k in range(1, 11)]
trainarate10 = [averageerrate(10, train, k, train) for k in range(1, 11)]
testarate10 =  [averageerrate(10, train, k, test) for k in range(1, 11)]           
trainarate15 = [averageerrate(15, train, k, train) for k in range(1, 11)]
testarate15 =  [averageerrate(15, train, k, test) for k in range(1, 11)]         
trainarate20 = [averageerrate(20, train, k, train) for k in range(1, 11)]
testarate20 =  [averageerrate(20, train, k, test) for k in range(1, 11)]         
trainarate25 = [averageerrate(25, train, k, train) for k in range(1, 11)]
testarate25 =  [averageerrate(25, train, k, test) for k in range(1, 11)]         
trainarate30 = [averageerrate(30, train, k, train) for k in range(1, 11)]
testarate30 =  [averageerrate(30, train, k, test) for k in range(1, 11)]  

k = range(1, 11) 
plt.plot(k, trainarate5) 
plt.plot(k, trainarate10)
plt.plot(k, trainarate15) 
plt.plot(k, trainarate20)
plt.plot(k, trainarate25)
plt.plot(k, trainarate30)
plt.xlabel('k') 
plt.ylim((0, 0.05))
plt.ylabel('train error rate')
plt.legend(['L = 5', 'L = 10', 'L = 15', 'L = 20', 'L = 25', 'L = 30'], loc='lower right')


k = range(1, 11)
plt.plot(k, testarate5) 
plt.plot(k, testarate10)
plt.plot(k, testarate15) 
plt.plot(k, testarate20)
plt.plot(k, testarate25)
plt.plot(k, testarate30)
plt.xlabel('k') 
plt.ylim((0, 0.06))
plt.ylabel('test error rate')
plt.legend(['L = 5', 'L = 10', 'L = 15', 'L = 20', 'L = 25', 'L = 30'], loc='lower right')












