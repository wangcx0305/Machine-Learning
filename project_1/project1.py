# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 20:33:26 2016

@author: wangchunxiao
"""

from pandas import read_csv
from statistics import mean, stdev
import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
from random import shuffle



train = read_csv('/Users/wangchunxiao/Desktop/cs534/project1/train.csv', header = None)
train = train.as_matrix()
for i in range(1, 45):
    train[:, i] = (train[:, i] - mean(train[:, i])) / stdev(train[:, i])
    
test = read_csv('/Users/wangchunxiao/Desktop/cs534/project1/test.csv', header = None)
test = test.as_matrix()
for i in range(1, 45):
    test[:, i] = (test[:, i] - mean(test[:, i])) / stdev(test[:, i])

#Part 1
    
x = np.matrix(train[:, 0:45])
y = np.matrix(train[:, 45]).T
w0 = np.matrix(np.append([0.01], np.zeros(44))).T

#set a fixed lambda = 0.1
lambda0 = 0.1

  ##batch gradient descent
wtrue = inv(lambda0 * np.matrix(np.identity(45)) + x.T * x) * x.T * y
alpha = np.arange(0, 0.001, 0.0001)
alpha = np.append(alpha, [10, 100, 1000])
alpha = [i for i in alpha if i != 0]
f = np.empty(shape = (500, len(alpha)))
f[:] = np.nan

for i in range(len(alpha)):
  grad = -2 * x.T * (y - x * w0) + 2 * lambda0 * w0
  step = 0
  w_new = w0
  while(norm(grad) > 1e-04 and step < 500):
    w_new = w_new - alpha[i] * grad
    f[step, i] = (y - x * w_new).T * (y - x * w_new) + lambda0 * w_new.T * w_new
    step = step + 1
    grad = -2 * x.T * (y - x * w_new) + 2 * lambda0 * w_new

fig = plt.figure()
    
ax1 = fig.add_subplot(431)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,0]]
ax1.plot(x,y)
ax1.set_xlabel('iter')
ax1.set_ylabel('r-sse')
ax1.set_title('alpha = 0.0001')

ax2 = fig.add_subplot(432)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,1]]
ax2.plot(x,y)
ax2.set_xlabel('iter')
ax2.set_ylabel('r-sse')
ax2.set_title('alpha = 0.0002')

ax3 = fig.add_subplot(433)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,2]]
ax3.plot(x,y)
ax3.set_xlabel('iter')
ax3.set_ylabel('r-sse')
ax3.set_title(' alpha = 0.0003')

ax4 = fig.add_subplot(434)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,3]]
ax4.plot(x,y)
ax4.set_xlabel('iter')
ax4.set_ylabel('r-sse')
ax4.set_title('alpha = 0.0004')

ax5 = fig.add_subplot(435)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,4]]
ax5.plot(x,y)
ax5.set_xlabel('iter')
ax5.set_ylabel('r-sse')
ax5.set_title('alpha = 0.0005')

ax6 = fig.add_subplot(436)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,5]]
ax6.plot(x,y)
ax6.set_xlabel('iter')
ax6.set_ylabel('r-sse')
ax6.set_title('alpha = 0.0006')

ax7 = fig.add_subplot(437)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,6]]
ax7.plot(x,y)
ax7.set_xlabel('iter')
ax7.set_ylabel('r-sse')
ax7.set_title('alpha = 0.0007')

ax8 = fig.add_subplot(438)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,7]]
ax8.plot(x,y)
ax8.set_xlabel('iter')
ax8.set_ylabel('r-sse')
ax8.set_title('alpha = 0.0008')

ax9 = fig.add_subplot(439)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,8]]
ax9.plot(x,y)
ax9.set_xlabel('iter')
ax9.set_ylabel('r-sse')
ax9.set_title('alpha = 0.0009')

ax10 = fig.add_subplot(4,3,10)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,9]]
ax10.plot(x,y)
ax10.set_xlabel('iter')
ax10.set_ylabel('r-sse')
ax10.set_title('alpha = 10')

ax11 = fig.add_subplot(4,3,11)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,10]]
ax11.plot(x,y)
ax11.set_xlabel('iter')
ax11.set_ylabel('r-sse')
ax11.set_title('alpha = 100')

ax12 = fig.add_subplot(4,3,12)
x = [i + 1 for i in range(500)]
y = [i for i in f[:,11]]
ax12.plot(x,y)
ax12.set_xlabel('iter')
ax12.set_ylabel('r-sse')
ax12.set_title('alpha = 1000')

fig.suptitle('regulized SSE plot with different alpha')

fig.tight_layout()



  ##stochastic  gradient descent
alpha = np.arange(0, 0.001, 0.0001)
alpha = [i for i in alpha if i != 0]
alpha = np.append(alpha, [0.1, 0.5, 1, 10, 100, 1000])
f = np.empty(shape = (5000, len(alpha)))
f[:] = np.nan


for i in range(len(alpha)):
    k = 0
    w_new = w0
    gra = -2 * x[0, :].T * (y[0] - x[0, :] * w_new) + 2 * lambda0 * w_new
    descent = alpha[i] * gra
    while(norm(descent) > 1e-06 and k < 5000):
        w_new = w_new -  descent
        f[k, i] = (y - x * w_new).T * (y - x * w_new) + lambda0 * w_new.T * w_new
        k = k + 1
        j = k % 100
        gra = -2 * x[j, :].T * (y[j] - x[j, :] * w_new) + 2 * lambda0 * w_new
        descent = alpha[i] * gra
    




#Part 2.a
vlambda0 = [0, 5* 1e-04, 1e-03, 5 * 1e-03, 1e-02, 5 * 1e-02, 1e-01, 5 * 1e-01, 1, 5, 10, \
            50, 100, 500, 1000, 5000]

hatw = [inv(lambda0 * np.identity(45) + x.T * x) * x.T * y for lambda0 in vlambda0]

trainsse = [(y - x * w).T * (y - x * w) for w in hatw]
trainsse = [a.item(0) for a in trainsse]
            
plt.plot(vlambda0, trainsse, 'o')   
plt.xlabel('$\lambda$')
plt.ylabel('SSE')
plt.title('SSE vs $\lambda$ for training data')
plt.show()

#part 2.b
xtest = np.matrix(test[:, 0:45])

ytest = np.matrix(test[:, 45]).T

testsse = [(ytest - xtest * w).T * (ytest - xtest * w) for w in hatw]
testsse = [a.item(0) for a in testsse]

plt.plot(vlambda0, testsse, 'o')   
plt.xlabel('$\lambda$')
plt.ylabel('SSE')
plt.title('SSE vs $\lambda$ for test data')
plt.show()

#Part 3
cvsse = list()
num = [x for x in range(100)]
shuffle(num)
for i in range(len(vlambda0)):
  sse = 0; 
  for j in range(10):
    ord = num[(10 * j) : (10 * j + 10)]
    left = [i for i in num if i not in ord]
    cvxtest = x[ord, :]
    cvytest = y[ord]
    cvxtrain = x[left, :]
    cvytrain = y[left]
    hatw = inv(vlambda0[i] * np.identity(45) + cvxtrain.T * cvxtrain) * cvxtrain.T * cvytrain
    psse = (cvytest - cvxtest * hatw).T * (cvytest - cvxtest * hatw)
    sse = sse + psse
  cvsse.append(sse)










