import numpy as np
import math
import random
import matplotlib.pyplot as plt
#from sklearn.linear_model import LogisticRegression,SGDClassifier
import copy
from scipy import linalg
import asgd
from loss import *

def obj(x, w):
    x = np.array(x)
    w = np.array(w)

    t = -sum(x * w)
  
    return 1.0 / (1 + np.exp(t))        

class ASGD:
    def __init__(self,x, y, loss = 'logloss', gamma = None, _lambda = 0.01, T = 1, a = None, c = None):
        """"""
        one = np.ones([len(x), 1])
        self._x = np.array(x)
        self._x = np.hstack([one, self._x])
        self._y = np.array(y)
        #self._v_x = np.hstack([np.ones([v_x.shape[0], 1]), v_x])
        #self._v_y = v_y
        self._m = self._x.shape[0]
        self._n = self._x[0].shape[0]
        self._loss_type = loss
        
        rand_id = range(0, self._m)
        random.shuffle(rand_id)
        
        if gamma == None:
            M = 0.0
            for i in rand_id[0: 1000]:
                r = linalg.norm(self._x[i, :], 2)
                if r > M:
                    M = r
            self._gamma0 = 1.0 / M
        else:
            self._gamma0 = gamma #learn rating
        if a == None:
            self._a = self._gamma0
        else:
            self._a = a
        if c == None:
            self._c = 3.0 / 4
        else:
            self._c = c
            
        self._lambda = _lambda
        self._T = T
        self._eta_t = 0.0
        
    def preproX(self,x, u, alfa):
        r = (1.0/ alfa) * u * x
        return r
    def learnRat(self, t):
        r = self._gamma0 * math.pow(1.0 + self._a * self._gamma0 * t, -self._c) 
        #r = 0.2 / math.sqrt(1 + t)
        #r = (1+0.02 * t) ** (-2.0/3)
        return r
    def loss(self, theta_t, v_x):
        r = 1.0 / (1 + np.exp(-theta_t.dot(v_x.transpose())))
        r[r > 0.5] = 1
        r[r < 0.5] = 0
        rat = 1.0 * sum(r != self._v_y) / v_x.shape[0]
        return rat
    
    def train(self):
        self.alfa_t = 1.0
        self.beta_t = 1.0
        self.tau_t = 0
        theta0 = np.zeros(self._n)
        for i in xrange(theta0.shape[0]):
            theta0[i] = random.random()*0.0
        self.u_t = theta0
        self.u1_t = theta0
        
        if self._loss_type == 'logloss':
            logloss = LogLoss()
            self._loss = logloss
        elif self._loss_type == 'hingeloss':
            hingloss = HingeLoss()
            self._loss = hingloss
        elif self._loss_type == 'squaredhingeloss':
            hingloss = SquaredHingeLoss()
            self._loss = hingloss
        elif self._loss_type == 'smoothhingeloss':
            hingloss = SmoothHingeLoss()
            self._loss = hingloss
        
        #train
        K = 1
        for itera in xrange(self._T):
            for i in xrange(0, self._m):
                xi = self._x[i, :]
                yi = self._y[i]
                self.trainOne(K, xi, yi)
                K = K + 1
        self._w = (self.tau_t * self.u_t + self.u1_t) / self.beta_t                
    def trainOne(self, K, xi, yi):
        #learning rate
        gamma_t = self.learnRat(K)
        eta_t = 1.0 / K
        xi_trans = (self.u_t / self.alfa_t).dot(xi)
        g_t = self._loss.dloss(xi_trans, yi) * xi
        self.alfa_t = float(self.alfa_t) / (1.0 - self._lambda * gamma_t)
        self.beta_t = float(self.beta_t) / (1.0 - (eta_t if eta_t != 1 else 0))            
        self.u_t = self.u_t - self.alfa_t * gamma_t * g_t

        self.u1_t = self.u1_t + self.tau_t * self.alfa_t * gamma_t * g_t
        self.tau_t = self.tau_t + 1.0 * eta_t * self.beta_t / self.alfa_t
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
            
        
def data(n1 = 500, n2 = 500, rat = 0.7):
    x = []
    y = []
    for i in xrange(n1):
        xi = []
        for j in xrange(30):
            xi.append(random.random() + 2)
        x.append(xi)
        y.append(1)
    
    for i in xrange(n2):
        xi = []
        for j in xrange(30):
            xi.append(random.random() + 1)
        x.append(xi)
        y.append(0)
    
    x = np.array(x)
    y = np.array(y)
    id = range(0, len(x))
    random.shuffle(id)
    x = x[id, :]
    y = y[id]
    writer = file('sgd_test.txt', 'w')
    for i in xrange(x.shape[0]):
        s = ''
        s = s + str(y[i]) + ' '
        for j in xrange(x[i].shape[0]):
            s = s + str(x[i][j]) + ' '
        writer.write(s[0: -1] + '\n')
    writer.close()
    

    index = int(x.shape[0] * rat)
    train_x = x[0: index, :]
    train_y = y[0: index]
    
    test_x = x[index + 1: x.shape[0], :]
    test_y = y[index + 1: x.shape[0]]
    
    return [train_x, train_y, test_x, test_y]
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
def testASGD():
    train_x, train_y, test_x, test_y = readData() #data(500,500)
    train_y = np.array(train_y)
    
    #np.random.seed(42)
    #n = 1000
    #p = 20
    #little_x = np.random.normal(0, 5, n*p).reshape((n, p))
    #w = np.random.normal(0, 2, p).reshape((p, 1))
    #little_y = little_x.dot(w) + np.random.normal(0, 0.5, n).reshape(n, 1)
    #little_y = little_y[:, 0 ]
    #little_y[little_y > 0] = 1
    #little_y[little_y < 0] = 0
    
    #train_x = little_x[0:800, :]
    #test_x = little_x[800:1000, :]
    #train_y = little_y[0:800]
    #test_y = little_y[800:1000]
    v_x = test_x[0:100,:]
    v_y = test_y[0:100]
    asgd = ASGD(train_x, train_y,loss = 'hingeloss', T = 1)
    asgd.train()
    #plt.plot(asgd._conv)
    #plt.show()
    pred_y = asgd.predictor(test_x[100: 300,:])
    r = test_y[100:300] == pred_y
    
    print test_y == pred_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = readData() #data(500,500)
    train_y = np.array(train_y)
    
    #np.random.seed(42)
    #n = 1000
    #p = 20
    #little_x = np.random.normal(0, 5, n*p).reshape((n, p))
    #w = np.random.normal(0, 2, p).reshape((p, 1))
    #little_y = little_x.dot(w) + np.random.normal(0, 0.5, n).reshape(n, 1)
    #little_y = little_y[:, 0 ]
    #little_y[little_y > 0] = 1
    #little_y[little_y < 0] = 0
    
    #train_x = little_x[0:800, :]
    #test_x = little_x[800:1000, :]
    #train_y = little_y[0:800]
    #test_y = little_y[800:1000]
    asgd = ASGD(train_x, train_y,loss = 'logloss',gamma = 1, T = 1)
    asgd.train()
    #plt.plot(asgd._conv)
    #plt.show()
    pred_y = asgd.predictor(test_x)
    r = test_y == pred_y
    
    print 'error rating = %f \n' % (1.0 * sum(r == False) / test_x.shape[0])

    
    
    
