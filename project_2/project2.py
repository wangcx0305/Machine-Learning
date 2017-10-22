#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:54:52 2016

@author: wangchunxiao
"""
from pandas import read_csv
import numpy as np
from heapq import nlargest
from matplotlib import pyplot as plt


######################################################################
######################################################################
######basic implementation############################################

datatrain = read_csv('wordtrain', header = None)
datatest = read_csv('wordtest', header = None)
labeltrain = read_csv('labeltrain', header = None)
labeltest = read_csv('labeltest', header = None)

traind = list()
for i in range(len(datatrain)):
    a = [int(x) for x in datatrain.iloc[i].str.split()[0]]
    traind.append(a)
       
trainin = list()
for i in range(len(labeltrain)):
    a =  labeltrain.iloc[i].str.split()[0]
    if a == ['HillaryClinton']:
        trainin.append(1)
    else:
        trainin.append(0) 
        
testd = list()
for i in range(len(datatest)):
    a = [int(x) for x in datatest.iloc[i].str.split()[0]]
    testd.append(a)
    
testin = list()
for i in range(len(labeltest)):
    a =  labeltest.iloc[i].str.split()[0]
    if a == ['HillaryClinton']:
        testin.append(1)
    else:
        testin.append(0) 
        

ph =  sum(x > 0 for x in trainin) / 5155
pt =  1 - ph


##split the train data in class Hillary and class Trump

trainh = list()
traint = list()
for i in range(len(traind)):
    if trainin[i] == 1:
        trainh.append(traind[i])
    else:
        traint.append(traind[i])
        

##Bernoulli Model 
##we use beta(2, 2) prior for smoothing, for a word never appeares in train data, 
##(2, 2), so p = 1 / (N_i + 2)

wordh = [x for sublist in trainh for x in sublist]
wordt = [x for sublist in traint for x in sublist]

uniwordh = list(set(wordh))
uniwordt = list(set(wordt))
        
def bp(word, can):
    if can == 1:
        num = 0
        for i in range(len(trainh)):
            if word in trainh[i]:
                num = num + 1;
        prob = (num + 1) / (len(trainh) + 2)
    else:
        num = 0
        for i in range(len(traint)):
            if word in traint[i]:
                num = num + 1;
        prob = (num + 1) / (len(traint) + 2)
    return prob
    
uniproh = [bp(word, 1) for word in uniwordh]
uniprot = [bp(word, 0) for word in uniwordt]

prepro = np.empty((len(testd), 2))
prepro[:] = np.nan  

for i in range(len(testd)):
   lprobh = 0;
   lprobt = 0;
   
   for j in range(len(testd[i])):
       if testd[i][j] in uniwordh:
           lprobh = lprobh + np.log(uniproh[uniwordh.index(testd[i][j])])
       else:
           lprobh = lprobh + np.log(1 / (len(trainh) + 2))
           
   for k in range(len(uniwordh)):
       if uniwordh[k] not in testd[i]:
           lprobh = lprobh + np.log(1 - uniproh[k])
           
   for j in range(len(testd[i])):
       if testd[i][j] in uniwordt:
           lprobt = lprobt + np.log(uniprot[uniwordt.index(testd[i][j])])
       else:
           lprobt = lprobt + np.log(1 / (len(traint) + 2))
   
   for k in range(len(uniwordt)):
       if uniwordt[k] not in testd[i]:
           lprobt = lprobt + np.log(1 - uniprot[k])
           
   prepro[i, 0] = lprobh + np.log(ph) 
   prepro[i, 1] = lprobt + np.log(pt)
       
prein = list()
for i in range(len(testd)):
    if prepro[i, 0] > prepro[i, 1]:
        prein.append(1)
    else:
        prein.append(0) 
        
prename = list()
for i in range(len(prein)):
    if(prein[i] == 1):
        prename.append('H')
    else:
        prename.append('T')
        
import csv
predictFile = open("biprediction.csv","w")
wr = csv.writer(predictFile, dialect = 'excel')
for item in prename:
    wr.writerow(item)
predictFile.close()        

       
bnaccu = list()

for i in range(len(testin)):
    if testin[i] == prein[i]:
        bnaccu.append(1)
    else:
        bnaccu.append(0)
        
bpaccu = sum(bnaccu) / len(testin)  
      
      
       
##multinomial model
####for a word never appeares in train data, we will use dirichlet prior
##with K parameter (2, 2, 2, ...), so p_ij = (n_ij + 1) / (n_i + K)   
     
nh = len(wordh)
nt = len(wordt)
nuwordh = list()
nuwordt = list()

for i in range(len(uniwordh)):
    num = 0
    for j in range(len(trainh)):
        num = num + trainh[j].count(uniwordh[i])
    nuwordh.append(num)
    
for i in range(len(uniwordt)):
    num = 0
    for j in range(len(traint)):
        num = num + traint[j].count(uniwordt[i])
    nuwordt.append(num)
     
wordtest = [x for sublist in testd for x in sublist] 
uniwordtest = list(set(wordtest))   
 
diffh = [word for word in uniwordtest if word not in uniwordh]      
difft = [word for word in uniwordtest if word not in uniwordt]

allwordh = uniwordh + diffh
allwordt = uniwordt + difft

Kh = len(allwordh)
Kt = len(allwordt)

allprobh = list()
allprobt = list()

for i in range(len(uniwordh)):
    allprobh.append((nuwordh[i] + 1) / (nh + Kh))

for i in range(len(diffh)):
    allprobh.append(1 / (nh + Kh))
    
for i in range(len(uniwordt)):
    allprobt.append((nuwordt[i] + 1) / (nt + Kt))

for i in range(len(difft)):
    allprobt.append(1 / (nt + Kt))

mprepro = np.empty((len(testd), 2))
mprepro[:] = np.nan

for i in range(len(testd)):
    lprobh = 0;
    lprobt = 0;
    
    for j in range(len(testd[i])):
        lprobh = lprobh + np.log(allprobh[allwordh.index(testd[i][j])])
        lprobt = lprobt + np.log(allprobt[allwordt.index(testd[i][j])])
    
    lprobh = lprobh + np.log(ph)
    lprobt = lprobt + np.log(pt)
    mprepro[i, 0] = lprobh
    mprepro[i, 1] = lprobt
    
mprein = list()
for i in range(len(testd)):
    if mprepro[i, 0] > mprepro[i, 1]:
        mprein.append(1)
    else:
        mprein.append(0)
    
mnaccu = list()
for i in range(len(testd)):
    if mprein[i] == testin[i]:
        mnaccu.append(1)
    else:
        mnaccu.append(0)
        
multiprename = list()
for i in range(len(mprein)):
    if(mprein[i] == 1):
        multiprename.append('H')
    else:
        multiprename.append('T')
        
import csv
predictFile = open("multiprediction.csv","w")
wr = csv.writer(predictFile, dialect = 'excel')
for item in prename:
    wr.writerow(item)
predictFile.close()        
      
mpaccu = sum(mnaccu) / len(testd)
    
##for bernoulli model to create 2 * 2 table
bmisch = 0
brigch = 0
bmisct = 0
brigct = 0

for i in range(len(testin)):
    if(testin[i] == 1 and prein[i] == 1):
        brigch = brigch + 1
        
    elif(testin[i] == 1 and prein[i] == 0):
            bmisch = bmisch + 1
            
    elif(testin[i] == 0 and prein[i] == 0):
                brigct = brigct + 1
                
    else:
                bmisct = bmisct + 1
    

## for multinomial model to create 2 * 2 table
mmisch = 0
mrigch = 0
mmisct = 0
mrigct = 0

for i in range(len(testin)):
    if(testin[i] == 1 and mprein[i] == 1):
        mrigch = mrigch + 1
        
    elif(testin[i] == 1 and mprein[i] == 0):
            mmisch = mmisch + 1
            
    elif(testin[i] == 0 and mprein[i] == 0):
                mrigct = mrigct + 1
                
    else:
                mmisct = mmisct + 1



##largest probability:
larnh = nlargest(10, nuwordh)
larph = [x / len(wordh) for x in larnh]
larwh = list()
larnt = nlargest(10, nuwordt)
larpt = [x / len(wordt) for x in larnt]
larwt = list()

for i in range(10):
    larwh.append(uniwordh[nuwordh.index(larnh[i])])
    larwt.append(uniwordt[nuwordt.index(larnt[i])])
    
voca = read_csv('voca', header = None)
voca = list(voca.values.flatten())
    
larrealwh = [voca[i] for i in larwh]       
larrealwt = [voca[i] for i in larwt]    
    
larrealwh
larrealwt 





   
    
#####################################################################
#####################################################################    
########Priors and overfitting#######################################


datatrain = read_csv('wordtrain', header = None)
datatest = read_csv('wordtest', header = None)
labeltrain = read_csv('labeltrain', header = None)
labeltest = read_csv('labeltest', header = None)

traind = list()
for i in range(len(datatrain)):
    a = [int(x) for x in datatrain.iloc[i].str.split()[0]]
    traind.append(a)
       
trainin = list()
for i in range(len(labeltrain)):
    a =  labeltrain.iloc[i].str.split()[0]
    if a == ['HillaryClinton']:
        trainin.append(1)
    else:
        trainin.append(0) 
        
testd = list()
for i in range(len(datatest)):
    a = [int(x) for x in datatest.iloc[i].str.split()[0]]
    testd.append(a)
    
testin = list()
for i in range(len(labeltest)):
    a =  labeltest.iloc[i].str.split()[0]
    if a == ['HillaryClinton']:
        testin.append(1)
    else:
        testin.append(0) 
        

ph =  sum(x > 0 for x in trainin) / 5155
pt =  1 - ph


##split the train data in class Hillary and class Trump

trainh = list()
traint = list()
for i in range(len(traind)):
    if trainin[i] == 1:
        trainh.append(traind[i])
    else:
        traint.append(traind[i])
        
wordh = [x for sublist in trainh for x in sublist]
wordt = [x for sublist in traint for x in sublist]

uniwordh = list(set(wordh))
uniwordt = list(set(wordt))

nh = len(wordh)
nt = len(wordt)
nuwordh = list()
nuwordt = list()
         
for i in range(len(uniwordh)):
    num = 0
    for j in range(len(trainh)):
        num = num + trainh[j].count(uniwordh[i])
    nuwordh.append(num)
    
for i in range(len(uniwordt)):
    num = 0
    for j in range(len(traint)):
        num = num + traint[j].count(uniwordt[i])
    nuwordt.append(num)
     
wordtest = [x for sublist in testd for x in sublist] 
uniwordtest = list(set(wordtest))   
 
diffh = [word for word in uniwordtest if word not in uniwordh]      
difft = [word for word in uniwordtest if word not in uniwordt]

allwordh = uniwordh + diffh
allwordt = uniwordt + difft

def accuracy(alpha):
  
    Kh = len(allwordh) * alpha 
    Kt = len(allwordt) * alpha
    
    allprobh = list()
    allprobt = list()
    
    for i in range(len(uniwordh)):
        allprobh.append((nuwordh[i] + alpha) / (nh + Kh))
    
    for i in range(len(diffh)):
        allprobh.append(alpha / (nh + Kh))
        
    for i in range(len(uniwordt)):
        allprobt.append((nuwordt[i] + alpha) / (nt + Kt))
    
    for i in range(len(difft)):
        allprobt.append(alpha / (nt + Kt))
    
    mprepro = np.empty((len(testd), 2))
    mprepro[:] = np.nan
    
    for i in range(len(testd)):
        lprobh = 0;
        lprobt = 0;
        
        for j in range(len(testd[i])):
            lprobh = lprobh + np.log(allprobh[allwordh.index(testd[i][j])])
            lprobt = lprobt + np.log(allprobt[allwordt.index(testd[i][j])])
        
        lprobh = lprobh + np.log(ph)
        lprobt = lprobt + np.log(pt)
        mprepro[i, 0] = lprobh
        mprepro[i, 1] = lprobt
        
    mprein = list()
    for i in range(len(testd)):
        if mprepro[i, 0] > mprepro[i, 1]:
            mprein.append(1)
        else:
            mprein.append(0)
        
    mnaccu = list()
    for i in range(len(testd)):
        if mprein[i] == testin[i]:
            mnaccu.append(1)
        else:
            mnaccu.append(0)
            
    mpaccu = sum(mnaccu) / len(testd)
    return mpaccu
        
        
    
alphalist = [1e-05, 1e-03, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
             0.6, 0.7, 0.8, 0.9, 1] 
pacculist = [accuracy(alpha) for alpha in alphalist]

logalpha = [np.log10(x) for x in alphalist]

plt.plot(logalpha, pacculist)
plt.xlabel('$log_{10}(alpha)$')
plt.ylabel('test set accuracy')
plt.title('accuracy versus alpha')








############################################################################
############################################################################
#####################Identifying important features#########################

#we explore the problem only under the multinomial model and use alpha = 1##
datatrain = read_csv('wordtrain', header = None)
datatest = read_csv('wordtest', header = None)
labeltrain = read_csv('labeltrain', header = None)
labeltest = read_csv('labeltest', header = None)

alpha = 1

traind = list()
for i in range(len(datatrain)):
    a = [int(x) for x in datatrain.iloc[i].str.split()[0]]
    traind.append(a)
       
trainin = list()
for i in range(len(labeltrain)):
    a =  labeltrain.iloc[i].str.split()[0]
    if a == ['HillaryClinton']:
        trainin.append(1)
    else:
        trainin.append(0) 
        
testd = list()
for i in range(len(datatest)):
    a = [int(x) for x in datatest.iloc[i].str.split()[0]]
    testd.append(a)
    
testin = list()
for i in range(len(labeltest)):
    a =  labeltest.iloc[i].str.split()[0]
    if a == ['HillaryClinton']:
        testin.append(1)
    else:
        testin.append(0) 
        

ph =  sum(x > 0 for x in trainin) / 5155
pt =  1 - ph


##split the train data in class Hillary and class Trump

trainh = list()
traint = list()
for i in range(len(traind)):
    if trainin[i] == 1:
        trainh.append(traind[i])
    else:
        traint.append(traind[i])
        
wordh = [x for sublist in trainh for x in sublist]
wordt = [x for sublist in traint for x in sublist]

uniwordh = list(set(wordh))
uniwordt = list(set(wordt))

nh = len(wordh)
nt = len(wordt)
nuwordh = list()
nuwordt = list()
         
for i in range(len(uniwordh)):
    num = 0
    for j in range(len(trainh)):
        num = num + trainh[j].count(uniwordh[i])
    nuwordh.append(num)
    
for i in range(len(uniwordt)):
    num = 0
    for j in range(len(traint)):
        num = num + traint[j].count(uniwordt[i])
    nuwordt.append(num)
     



commonword = [word for word in uniwordh if word in uniwordt]
leftwh = [word for word in uniwordh if word not in commonword]
leftwt = [word for word in uniwordt if word not in commonword]
pcoword = np.empty((len(commonword), 2))
pcoword[:] = np.nan
for i in range(len(commonword)):
    pcoword[i, 0] = np.log(nuwordh[uniwordh.index(commonword[i])] / len(wordh))
    pcoword[i, 1] = np.log(nuwordt[uniwordt.index(commonword[i])] / len(wordt))
    
division = [pcoword[i, 0] / pcoword[i, 1] for i in range(len(commonword))]
ldivision = [np.log(division[i]) for i in range(len(commonword))]             
 
###percent is the left percentage of the common word in terms of higher test power            
def adjustaccuracy(percent):
    upbound = np.percentile(division, 100 - percent / 2)
    lowbound = np.percentile(division, percent / 2)
    leftd = [d for d in division if d >= upbound or d <= lowbound]
    leftw = [commonword[division.index(d)] for d in leftd ]
    unipowerwh = leftw + leftwh
    unipowerwt = leftw + leftwt
    
    powerwordh = [x for x in wordh if x in unipowerwh]
    powerwordt = [x for x in wordt if x in unipowerwt]
    
    powernh = len(powerwordh)
    powernt = len(powerwordt)
    nupowerwh = list()
    nupowerwt = list()
             
    for i in range(len(unipowerwh)):
        num = 0
        for j in range(len(trainh)):
            num = num + trainh[j].count(unipowerwh[i])
        nupowerwh.append(num)
        
    for i in range(len(unipowerwt)):
        num = 0
        for j in range(len(traint)):
            num = num + traint[j].count(unipowerwt[i])
        nupowerwt.append(num)
         
    wordtest = [x for sublist in testd for x in sublist] 
    uniwordtest = list(set(wordtest))   
     
    diffh = [word for word in uniwordtest if word not in unipowerwh]      
    difft = [word for word in uniwordtest if word not in unipowerwt]
    
    allpowerwordh = unipowerwh + diffh
    allpowerwordt = unipowerwt + difft

    Kh = len(allpowerwordh) * alpha 
    Kt = len(allpowerwordt) * alpha
    
    allprobh = list()
    allprobt = list()
    
    for i in range(len(unipowerwh)):
        allprobh.append((nupowerwh[i] + alpha) / (powernh + Kh))
    
    for i in range(len(diffh)):
        allprobh.append(alpha / (powernh + Kh))
        
    for i in range(len(unipowerwt)):
        allprobt.append((nupowerwt[i] + alpha) / (powernt + Kt))
    
    for i in range(len(difft)):
        allprobt.append(alpha / (powernt + Kt))
    
    mprepro = np.empty((len(testd), 2))
    mprepro[:] = np.nan
    
    for i in range(len(testd)):
        lprobh = 0;
        lprobt = 0;
        
        for j in range(len(testd[i])):
            lprobh = lprobh + np.log(allprobh[allpowerwordh.index(testd[i][j])])
            lprobt = lprobt + np.log(allprobt[allpowerwordt.index(testd[i][j])])
        
        lprobh = lprobh + np.log(ph)
        lprobt = lprobt + np.log(pt)
        mprepro[i, 0] = lprobh
        mprepro[i, 1] = lprobt
        
    mprein = list()
    for i in range(len(testd)):
        if mprepro[i, 0] > mprepro[i, 1]:
            mprein.append(1)
        else:
            mprein.append(0)
        
    mnaccu = list()
    for i in range(len(testd)):
        if mprein[i] == testin[i]:
            mnaccu.append(1)
        else:
            mnaccu.append(0)
            
    mpaccu = sum(mnaccu) / len(testd)
    return mpaccu
        
percentlist = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 
pacculist = [adjustaccuracy(x) for x in percentlist]     

plt.plot(percentlist, pacculist)
plt.xlabel('percentage of left common words')
plt.ylabel('test set accuracy')
plt.title('accuracy versus percentage')













