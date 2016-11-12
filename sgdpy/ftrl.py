
# coding=utf-8 
import numpy as np
from loss import *
import matplotlib.pyplot as plt
import random

'''
ref:
<<Ad Click Prediction: a View from the Trenches>>
http://www.wbrecom.com/?p=412
'''
class FTRL:
    
    def __init__(self, x, y, alfa, beta, lamda1, lamda2,loss = 'logloss', iterat = 1):
        self._x = x
        self._y = y
        #one = np.ones([len(x), 1])
        #self._x = np.hstack([one, self._x])
        
        self._alfa = alfa
        self._beta = beta
        self._lamda1 = lamda1
        self._lamda2 = lamda2
        self._M = self._x.shape[0]
        self._N = self._x.shape[1]
        
        self._z = np.zeros(self._N)
        self._q = np.zeros(self._N)
        self._w = np.zeros(self._N)
        #self._w = np.random.normal(0, 0.5, self._N)
        
        self._iterat = iterat
        self._loss_type = loss
        
        if self._loss_type == 'logloss':
            logloss = LogLoss()
            self._loss = logloss
        elif self._loss_type == 'hingeloss':
            hingeloss = HingeLoss()
            self._loss = hingeloss
    
    def train(self):
        for t in xrange(self._iterat):
            print 'iter: ', t
            for j in xrange(self._M):
                xi = self._x[j]
                yi = self._y[j]
                wTx = self._w.dot(xi)
                g = self._loss.dloss(wTx, yi) * xi
                sigma = (np.sqrt(self._q + g * g) - np.sqrt(self._q)) / self._alfa
                self._q = self._q + g * g
                self._z = self._z + g - sigma * self._w
                self._w = [0.0 if abs(self._z[i]) < self._lamda1 
                     else -1.0 / (self._lamda2 + (self._beta + np.sqrt(self._q[i])) / self._alfa) * 
                     (self._z[i] - self._lamda1 * np.sign(self._z[i])) for i in xrange(self._N)]
                self._w = np.array(self._w)
                
    def predictor(self, x):
        y = []
        p = []
        x = np.array(x)
        for i in xrange(x.shape[0]):
            #xi = np.hstack([1,x[i, :]])
            xi = x[i, :]
            if self._loss_type == 'logloss':
                pi = self._loss.deci(self._w, xi)
                y.append(1 if pi > 0.5 else -1)
            elif self._loss_type == 'hingeloss' or self._loss_type == 'squaredhingeloss' or self._loss_type == 'smoothhingeloss':
                pi = self._loss.deci(self._w, xi)
                y.append(1 if pi > 0 else -1)
            p.append(pi)
        
        return y       
def readData(index = 700):
    y = []
    x = []
    for line in file('sgd_test.txt'):
        ele = line.split(' ')
        if int(ele[0]) == 0:
            y.append(-1)
        else:
            y.append(int(ele[0]))
        xi = []
        for e in ele[1: len(ele)]:
            xi.append(float(e))
        x.append(xi)
    
    x = np.array(x)
    #x = np.hstack([x[:, 0:5], x])
    y = np.array(y)
    
    train_x = x[0: index, :]
    train_y = y[0: index]
    
    test_x = x[index + 1: x.shape[0], :]
    test_y = y[index + 1: x.shape[0]]
    
    return [train_x, train_y, test_x, test_y] 
def writeLRData(f, train_x, train_y):
    writer = file(f, 'w')
    for i in xrange(len(train_x)):
        yi = train_y[i]
        xi = train_x[i]
        writer.write(str(yi))
        for j in xrange(len(xi)):
            if xi[j] != 0.0:
                writer.write(' ' + str(j) + ':' + str(xi[j]))
        writer.write('\n')
    writer.close()


def minstData():
    train_x = []
    train_y = []
    write = file('dumy.txt', 'w')
    for line in file(unicode('./mnist_train.txt','utf8'),'r'):
        ele = line.strip().split(' ')
        if int(ele[0]) == 8:
            train_y.append(1)
            xi = []
            for ei in ele[1: len(ele)]:
                xi.append(float(ei))
            train_x.append(xi)
            write.write('1 ' + ' '.join(ele[1: len(ele)]) + '\n')
        elif int(ele[0]) == 9:
            train_y.append(-1)
            xi = []
            for ei in ele[1: len(ele)]:
                xi.append(float(ei))
            train_x.append(xi)
            write.write('-1 ' + ' '.join(ele[1: len(ele)]) + '\n')
    write.close()
    test_x = []
    test_y = []
    for line in file(unicode('./mnist_test.txt', 'utf8'), 'r'):
        ele = line.strip().split(' ')
        if int(ele[0]) == 8:
            test_y.append(1)
            xi = []
            for ei in ele[1: len(ele)]:
                xi.append(float(ei))
            test_x.append(xi)
        elif int(ele[0]) == 9:
            test_y.append(-1)
            xi = []
            for ei in ele[1: len(ele)]:
                xi.append(float(ei))
            test_x.append(xi)
    print 'train sample ', len(train_x)
    writeLRData('mnist_train01.txt', train_x, train_y)
    writeLRData('mnist_test01.txt', test_x, test_y)
    return [train_x, train_y, test_x, test_y] 
def libSVMFormat(x, y):
    writer = file('libsvm_train', 'w')
    writer2 = file('libsvm_test', 'w')
    x1 = []
    x2 = []
    for i in xrange(100):
        if y[i] == 1:
            x1.append(x[i, 0:2])
        else:
            x2.append(x[i, 0: 2])
        writer.write(str(y[i]) + " ")
        writer.write('0:' + str(x[i][0]) + ' ' + '1:' + str(x[i][1]) + '\n')
    writer.close()
    
    for i in xrange(100, 150):
        writer2.write(str(y[i]) + " ")
        writer2.write('0:' + str(x[i][0]) + ' ' + '1:' + str(x[i][1]) + '\n')
    writer2.close()
    
if __name__ == '__main__':
    #train_x, train_y, test_x, test_y = readData()
    #libSVMFormat(train_x, train_y)
    
    train_x, train_y, test_x, test_y = minstData()
    
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    ftrl = FTRL(train_x, train_y, 1.0, 1.0, 0.1, 0.0,loss = 'logloss', iterat = 1)
    ftrl.train()
    #plt.plot(asgd._conv)
    #plt.show()
    pred_y = ftrl.predictor(test_x)
    r = np.array(test_y) == np.array(pred_y)
    
    print 'error rating = %f \n' % (1.0 * sum(r == False) / test_x.shape[0])
    
    
    
    
    
    
    
    
    
    
    
    
    
    