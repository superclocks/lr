import numpy as np
from loss import *
import matplotlib.pyplot as plt
import random
class AdaDelta:
    def __init__(self,x, y, loss = 'logloss', rho = 0.1, const = 0.0001,iter = 10):
        self._x = np.array(x)
        self._y = np.array(y)
        one = np.ones([len(x), 1])
        self._x = np.hstack([one, self._x])

        self._m, self._n = self._x.shape #the number of samples and the number of features
        self._loss_type = loss 
        self._iter = iter #the number of computer
        #self._lambda = lambd
        self._w = np.random.normal(0, 0.5, self._n)
        #self._w = np.zeros(self._n)
        self._rho = rho
        self._const = const
        self._delta_g = 0.0
        self._delta_omega = 0.0
        
        if self._loss_type == 'logloss':
            logloss = LogLoss()
            self._loss = logloss
        elif self._loss_type == 'hingeloss':
            hingeloss = HingeLoss()
            self._loss = hingeloss
    def trainOne(self, xi, yi):
        
        wTx = self._w.dot(xi)
        gt = self._loss.dloss(wTx, yi) * xi
        self._delta_g = self._rho * self._delta_g + (1.0 - self._rho) * (gt ** 2)
        rsm_gt = np.sqrt(self._delta_g + self._const)
        rsm_delta_omega = np.sqrt(self._delta_omega + self._const)
        
        delta_omega = -(rsm_delta_omega / rsm_gt) * gt
        self._delta_omega = self._rho * self._delta_omega + (1 - self._rho) * (delta_omega ** 2)
        self._w = self._w + delta_omega
        
    def train(self):
        for k in xrange(self._iter):
            for i in xrange(self._m):
                self.trainOne(self._x[i, :], self._y[i])
    def predictor(self, x):
        y = []
        p = []
        x = np.array(x)
        for i in xrange(x.shape[0]):
            xi = np.hstack([1,x[i, :]])
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
def data(n = 500, rat = 0.7):
    
    p = 300
    x1 = np.random.normal(0, 1, n*p).reshape((n, p))
    x2 = np.random.normal(3, 1, n*p).reshape((n, p))
    x = np.vstack([x1, x2])

    id = np.arange(0, x.shape[0])
    random.shuffle(id)
    x = x[id, :]
    
    w = np.random.normal(5, 0.5, p).reshape((p, 1))
    y = x.dot(w) + np.random.normal(0, 0.5, 2*n).reshape(2*n, 1)
    y_mean = y.mean()
    y = [1 if i>y_mean else -1 for i in y]
    y = np.array(y)
    index = int(x.shape[0] * rat)
    train_x = x[0: index, :]
    train_y = y[0: index]
    
    test_x = x[index + 1: x.shape[0], :]
    test_y = y[index + 1: x.shape[0]]
    
    return [train_x, train_y, test_x, test_y]
if __name__ == '__main__':
    #train_x, train_y, test_x, test_y = readData() #data(500,500)
    train_x, train_y, test_x, test_y = data()
    train_y = np.array(train_y)
    adadelta = AdaDelta(train_x, train_y,loss = 'logloss', rho = 0.9, const = 0.0001,iter = 5)
    adadelta.train()
    #plt.plot(asgd._conv)
    #plt.show()
    pred_y = adadelta.predictor(test_x)
    r = test_y == pred_y
    
    print 'error rating = %f \n' % (1.0 * sum(r == False) / test_x.shape[0])

    