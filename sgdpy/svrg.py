import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,SGDClassifier
import copy
from scipy import linalg
import asgd
from loss import *

def obj(x, w):
    x = np.array(x)
    w = np.array(w)

    t = -sum(x * w)
    #if t > 0:
        #try:
            #val = math.exp(-sum(x * w))
        #except Exception:
            #val = 1e20
    #elif t<0:
        #try:
            #val = math.exp(-sum(x * w))
        #except Exception:
            #val = 1e-20
    #else:
        #val = 1.0
    return 1.0 / (1 + np.exp(t))
    
def grad(x, y, w):
    x = np.array(x)
    w = np.array(w)
    t = -sum(x * w)
    if t > 0:
        try:
            val = math.exp(-sum(x * w))
        except Exception:
            val = 10000000000
    elif t<0:
        try:
            val = math.exp(-sum(x * w))
        except Exception:
            val = 0.0000000001
    else:
        val = 1.0
    r = (1.0 / (1 + val) - y) * x      
    return r
def agrad(x, y, w, alfa):
    #x1 = np.array(x1)
    #w = np.array(w)
    #pi = obj(x1, w)
    
    #r = y * math.log(pi, 2) + (1.0 - y) * math.log(1.0 - pi, 2)
    #rr = r * x1
    #return -rr
    x = np.array(x)
    w = w / alfa
    t = -sum(x * w)
    if t > 0:
        try:
            val = math.exp(-sum(x * w))
        except Exception:
            val = 10000000000
    elif t<0:
        try:
            val = math.exp(-sum(x * w))
        except Exception:
            val = 0.0000000001
    else:
        val = 1.0
    r = (1.0 / (1 + val) - y) * (x)    
    return r

def lassGrad(x, y, w, lam = 1):
    x = np.array(x)
    w = np.array(w)
    t = -sum(x * w)
    if t > 0:
        try:
            val = math.exp(-sum(x * w))
        except Exception:
            val = 10000000000
    elif t<0:
        try:
            val = math.exp(-sum(x * w))
        except Exception:
            val = 0.0000000001
    else:
        val = 1.0
    r = (1.0 / (1 + val) - y) * x - (lam / 700.0) * np.sign(w)  
    #gi = r * 4.0
    #yita = -np.array(max(list(-w), list(-gi)))
    return r

def numGrad(x, y, w):
    x = np.array(x)
    w = np.array(w)
    r = []    
    for i in xrange(w.shape[0]):
        wi = copy.deepcopy(w);
        wi[i] = w[i] + 0.000001
        r.append((obj(x, wi) - obj(x, w)) / 0.000001)      
    return np.array(r)
        
########################################################################
class SVRG:
    """"""

    #----------------------------------------------------------------------
    def __init__(self, x, y, ita = 0.1, k = 50, _lambda = 1,w = None):
        """Constructor"""
        one = np.ones([len(x), 1])
        self._x = np.array(x)
        self._x = np.hstack([ one,self._x])
        self._y = np.array(y)
        self._ita = ita
        self._k = k
        self._lambda = _lambda
        self._m = self._x.shape[0]
        
        if(w == None):
            self._w = []
            for i in xrange(self._x.shape[1]):
                self._w.append(random.random() * 0.0)
            self._w = np.array(self._w)
        else:
            self._w = w
    def gradMean(self, grad):
        mu = 0
        for i in xrange(self._x.shape[0]):
            mu = mu + grad(self._x[i, :], self._y[i], self._w)
        return mu / self._x.shape[0]
    
    def svrgTrain(self, obj, grad):
        self._conv = []
        w0 = copy.deepcopy(self._w)
        for s in xrange(self._k):
            mu = self.gradMean(grad)
            K = 1
            for t in xrange(self._m):
                it = random.randint(0, self._x.shape[0] - 1)
                xi = self._x[it, :]
                yi = self._y[it]
                w0 = w0 - (self._ita/math.sqrt(1+K)) * (grad(xi, yi, w0) - grad(xi, yi, self._w) + mu)
            K = K + 1
            self._conv.append(linalg.norm(w0 - self._w,2))
            self._w = copy.deepcopy(w0)
    def sgdTrain(self, obj, grad):
        w0 = self._w
        K=1
        aw = w0
        self._conv = []
        for s in xrange(self._k):   
            
            for t in xrange(self._x.shape[0]):
                #it = random.randint(0, self._x.shape[0] - 1)
                xi = self._x[t, :]
                yi = self._y[t]
                g = grad(xi, yi, w0)
                w0 = w0 - (self._ita/math.sqrt(1+K)) * g #
                self._w = (self._w + w0)
                K = K + 1
            self._conv.append(linalg.norm(aw - self._w / K,2))
            aw = self._w / K
            #self._w = w0
        self._w = self._w / K
    #SGD-L1 (Naive)¡±.    
    def sgdL1NaiveTrain(self, obj, grad):
        w0 = copy.deepcopy(self._w)
        self._conv = []
        for s in xrange(self._k):   
            K=1
            
            for t in xrange(self._x.shape[0]):
                #it = random.randint(0, self._x.shape[0] - 1)
                xi = self._x[t, :]
                yi = self._y[t]
                g = grad(xi, yi, w0)
                w0 = w0 - self._ita / math.sqrt(1+K) * (g + (self._lambda / self._m) * np.sign(w0))
                K=K+1
            self._conv.append(linalg.norm(w0 - self._w,2))
            #self._w = w0
            self._w = copy.deepcopy(w0)    
    #SGDL1(Clipping)
    def sgdL1ClippingTrain(self, obj, grad):
        w0 = copy.deepcopy(self._w)
        self._conv = []
        for s in xrange(self._k):   
            K=1
            for t in xrange(self._x.shape[0]):
                #it = random.randint(0, self._x.shape[0] - 1)
                xi = self._x[t, :]
                yi = self._y[t]
                g = grad(xi, yi, w0)
                w0 = w0 - (self._ita/math.sqrt(1+K)) * g
                for i in xrange(g.shape[0]):
                    if w0[i] > 0:
                        w0[i] = max(0, w0[i] - self._ita/math.sqrt(1+K) * (self._lambda / self._m))
                    elif w0[i] < 0:
                        w0[i] = min(0, w0[i] + self._ita/math.sqrt(1+K) * (self._lambda / self._m))                
                K=K+1
            self._conv.append(linalg.norm(w0 - self._w,2))  
            self._w = copy.deepcopy(w0)
        self._w = w0
    #SGD-L1 (Cumulative)
    def sgdL1CumulativeUpdateWeignt(self, xi, yi, grad, K):
        self.w0 = self.w0 - (self._ita / math.sqrt(1 + K)) * grad(xi, yi, self.w0)
        self.sgdL1CumulativeApplyPenalty()
    def sgdL1CumulativeApplyPenalty(self):
        z = copy.deepcopy(self.w0)
        for i in xrange(self._x[0].shape[0]):
            if self.w0[i] > 0:
                self.w0[i] = max(0, self.w0[i] - (self.u + self.q[i]))
            elif self.w0[i] < 0:
                self.w0[i] = min(0, self.w0[i] + (self.u - self.q[i]))
        self.q = self.q + self.w0 - z
        
    def sgdL1CumulativeTrain(self, obj, grad):
        self.w0 = copy.deepcopy(self._w)
        self.u = 0
        self.q = np.zeros(len(self.w0))
        self._conv = []
        for s in xrange(self._k):   
            K=1
            self.u = self.u + (self._ita / math.sqrt(1 + K)) * self._lambda / self._m
            
            for t in xrange(self._x.shape[0]):
                #it = random.randint(0, self._x.shape[0] - 1)
                xi = self._x[t, :]
                yi = self._y[t]
                self.sgdL1CumulativeUpdateWeignt(xi, yi, grad, K)
                K=K+1
            #self._w = w0
            self._conv.append(linalg.norm(self.w0 - self._w,2))
            self._w = copy.deepcopy(self.w0)
    def batchTrain(self, obj, grad):
        w0 = self._w
        for s in xrange(self._k): 
            wt = 0
            for t in xrange(self._x.shape[0]):
                #it = random.randint(0, self._x.shape[0] - 1)
                xi = self._x[t, :]
                yi = self._y[t]
                g = grad(xi, yi, w0)
                wt = wt + g
            w0 = w0 - (1./ 700) * self._ita * wt
            #self._w = w0
        self._w = w0
    def predictor(self, x):
        y = []
        p = []
        x = np.array(x)
        for i in xrange(x.shape[0]):
            xi = np.hstack([1,x[i, :]])
            pi = obj(xi, self._w)
            p.append(pi)
            if pi > 0.5:
                y.append(1)
            else:
                y.append(0)
        return y
    

class ASGD:
    #----------------------------------------------------------------------
    def __init__(self, obj, grad, x, y, v_x, v_y, gamma = None, _lambda = 0.01, T = 1, a = None, c = None):
        """"""
        one = np.ones([len(x), 1])
        self._x = np.array(x)
        self._x = np.hstack([one, self._x])
        self._y = np.array(y)
        self._v_x = np.hstack([np.ones([v_x.shape[0], 1]), v_x])
        self._v_y = v_y
        self._m = self._x.shape[0]
        self._n = self._x[0].shape[0]
        
        self._obj = obj
        self._grad = grad
        
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
    def sgdTrain(self, obj, grad):
        self._w = np.zeros(self._n)
        w0 = self._w
        K=1
        aw = w0
        self._conv = []
        for s in xrange(1):   
            
            for t in xrange(self._x.shape[0]):
                #it = random.randint(0, self._x.shape[0] - 1)
                xi = self._x[t, :]
                yi = self._y[t]
                g = grad(xi, yi, w0)
                w0 = w0 - (0.02/math.sqrt(1+K)) * g #
                self._w = (self._w + w0)
                K = K + 1
            self._conv.append(linalg.norm(aw - self._w / K,2))
            aw = self._w / K
            #self._w = w0
        self._w = self._w / K
        
    def preproX(self,x, u, alfa):
        r = (1.0/ alfa) * u * x
        return r
    def learnRat(self, t):
        r = self._gamma0 * math.pow(1.0 + self._a * self._gamma0 * t, -self._c) 
        #r = 0.2 / math.sqrt(1 + t)
        #r = (1+0.02 * t) ** (-2.0/3)
        return r
    def trainUpdate(self):
        self._conv = []
        alfa0 = 1.0
        beta0 = 1.0
        tau0 = 0
        theta0 = np.zeros(self._n)
        for i in xrange(theta0.shape[0]):
            theta0[i] = random.random()
        u0 = theta0
        u01 = theta0
        eta_t = 0.0
        self._w = u0
        t = 0.0
        for i in xrange(self._T):
            for j in xrange(1, self._m): 
                gamma_t = self.learnRat(t)
                #eta_t = eta_t + gamma_t
                if t <= self._m:
                    eta_t = 1.0
                else:
                    eta_t = 1.0 / (1.0 + (t - self._m))
                eta_t = 1.0 
                #pre_xi = self.preproX(self._x[j, :], u0, alfa0)
                g_t = agrad(self._x[j - 1, :], self._y[j -1], u0, alfa0)
                alfa_t = float(alfa0) / (1.0 - self._lambda * gamma_t)
                
                etd = alfa_t * gamma_t * sum(g_t / self._x[j - 1])
                if etd != 0.0:
                    u_t = u0 + etd * self._x[j - 1,:]
                    u0 = u_t
                if eta_t >= 1.0:
                    u1_t = np.zeros(self._n)
                    beta_t = alfa_t
                    tau_t = 1.0
                elif eta_t > 0:
                    if etd != 0.0:
                        u1_t = u01 - tau0 * alfa_t * gamma_t * g_t
                    beta_t = float(beta0) / (1.0 - 1.0 * eta_t)
                    tau_t = tau0 + (1.0 * eta_t) * beta_t / alfa_t
                #update
                #self._conv.append(linalg.norm(u0 - u_t,2))
                alfa0 = alfa_t
                beta0 = beta_t
                
                u01 = u1_t
                tau0 = tau_t
                t = t + 1.0
            
            self._w = (tau_t * u_t + u1_t) / beta_t
    def loss(self, theta_t, v_x):
        r = 1.0 / (1 + np.exp(-theta_t.dot(v_x.transpose())))
        r[r > 0.5] = 1
        r[r < 0.5] = 0
        rat = 1.0 * sum(r != self._v_y) / v_x.shape[0]
        return rat
    def train(self):
        self._conv = []
        alfa_t = 1.0
        beta_t = 1.0
        tau_t = 0
        theta0 = np.zeros(self._n)
        for i in xrange(theta0.shape[0]):
            theta0[i] = random.random()*0.0
        u_t = theta0
        u1_t = theta0
        eta_t = 0.0
        self._w = u_t
        K = 1
        ww = theta0
        theta1_t = np.zeros(self._n)

        self._loss = LogLoss()
        for i in xrange(self._T):
            for t in xrange(1, self._m + 1): 
                gamma_t = self.learnRat(K)
                #eta_t = 1 if K<self._m else 1.0 / K
                eta_t = 1.0 / K
                #pre_xi = self.preproX(self._x[t, :], u0, alfa0)
                g_t = agrad(self._x[t - 1, :], self._y[t -1], u_t, alfa_t)
                
                #xi_trans = (u_t / alfa_t).dot(self._x[t - 1])
                #g_t = -self._loss.dloss(xi_trans, self._y[t - 1]) * self._x[t - 1]
                
                
                alfa_t = float(alfa_t) / (1.0 - self._lambda * gamma_t)
                beta_t = float(beta_t) / (1.0 - (eta_t if eta_t != 1 else 0))            
                u_t = u_t - alfa_t * gamma_t * g_t

                theta_t = u_t / alfa_t
                theta1_t = 0.99 * theta1_t + 0.01 * theta_t
                
                rat = self.loss(theta_t, self._v_x)
                rat1 = self.loss(theta1_t, self._v_x)
                
                #if rat == 0:
                    #self._w = theta_t
                    ##return
                #elif rat1 == 0:
                    #self._w = theta1_t
                    ##return
                
                u1_t = u1_t + tau_t * alfa_t * gamma_t * g_t
                tau_t = tau_t + (1.0 * eta_t) * beta_t / alfa_t
                #update
                self._conv.append(linalg.norm((tau_t * u_t + u1_t) / (beta_t) - ww,2))
                alfa0 = alfa_t
                beta0 = beta_t

                K = K + 1
                ww = (tau_t * u_t + u1_t) / (beta_t)
                rat2 = self.loss(ww, self._v_x)
                print 'rat = %f, rat1 = %f, rat2 = %f\n' %(rat, rat1, rat2)
            self._w = (tau_t * u_t + u1_t) / (beta_t)
            #self._w = u0
    def predictor(self, x):
        y = []
        p = []
        x = np.array(x)
        for i in xrange(x.shape[0]):
            xi = np.hstack([1,x[i, :]])
            pi = obj(xi, self._w)
            p.append(pi)
            if pi > 0.5:
                y.append(1)
            else:
                y.append(0)
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
def testSGD():
    train_x, train_y, test_x, test_y = readData() #data(500,500)
    train_y = np.array(train_y)
    train_y = train_y.transpose()
    clf_l1_LR = SGDClassifier(loss="log",average=True, n_iter=1)
    clf_l1_LR.fit(train_x, train_y)
    pred_y = clf_l1_LR.predict(test_x)
    test_y = np.array(test_y)
    #SVRG train
    svrg = SVRG(train_x, train_y, ita = 0.2, k = 10, _lambda = 0.1)
    svrg.svrgTrain(obj, grad)
    pred_y = svrg.predictor(test_x)
    pred_y = np.array(pred_y)
    #general sgd train
    svrg1 = SVRG(train_x, train_y, ita = 0.2, k = 10, _lambda = 0.1)
    svrg1.sgdTrain(obj, grad)
    pred_y = svrg1.predictor(test_x)
    
    svrg2 = SVRG(train_x, train_y, ita = 0.2, k = 10, _lambda = 0.1)
    svrg2.sgdL1ClippingTrain(obj, grad)
    
    svrg3 = SVRG(train_x, train_y, ita = 0.1, k = 10, _lambda = 0.1)
    svrg3.sgdL1CumulativeTrain(obj, grad)
    
    fig = plt.figure(2)
    ax = fig.add_subplot(211)
    ax.plot(svrg1._conv, 'r')
    
    #ax = fig.add_subplot(312)
    ax.plot(svrg2._conv, 'g')
    
    #ax = fig.add_subplot(313)
    ax.plot(svrg3._conv)
    ax = fig.add_subplot(212)
    ax.plot(svrg._conv)
    plt.show()
    
    pred_y1 = svrg1.predictor(test_x)
    pred_y1 = np.array(pred_y1)
    fig = plt.figure()
    ax = fig.add_subplot(411)
    ax.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1],'r.')
    ax.plot(train_x[train_y == 0, 0], train_x[train_y == 0, 1],'ro')
    
    ax = fig.add_subplot(412)
    ax.plot(test_x[test_y == 1, 0], test_x[test_y == 1, 1],'r.')
    ax.plot(test_x[test_y == 0, 0], test_x[test_y == 0, 1],'ro')
    
    plt.title('SVRG: four steps converge')
    ax = fig.add_subplot(413)
    ax.plot(test_x[pred_y == 1, 0], test_x[pred_y == 1, 1],'r.')
    ax.plot(test_x[pred_y == 0, 0], test_x[pred_y == 0, 1],'ro')
    
    ax = fig.add_subplot(414)
    ax.plot(test_x[pred_y1 == 1, 0], test_x[pred_y1 == 1, 1],'r.')
    ax.plot(test_x[pred_y1 == 0, 0], test_x[pred_y1 == 0, 1],'ro')
    plt.show()
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
    asgd = ASGD(obj, grad,train_x, train_y, v_x, v_y, T = 10)
    asgd.train()
    plt.plot(asgd._conv)
    plt.show()
    pred_y = asgd.predictor(test_x[100: 300,:])
    r = test_y[100:300] == pred_y
    
    print test_y == pred_y
def testAsgd():
    train_x, train_y, test_x, test_y = readData() #data(500,500)
    train_y = np.array(train_y)
    from asgd import *
    asgd = NaiveBinaryASGD(train_x.shape[0])
    asgd.partial_fit(train_x, train_y)
    pred_y = asgd.predictor(test_x)
    
def fineDiff():
    
    for j in xrange(699,700):
        train_x, train_y, test_x, test_y = readData(j) #data(500,500)
        train_y = np.array(train_y)
        
        
        asgd1 = asgd.ASGD(10)
        asgd1.fit(train_x, train_y)
        w1 = asgd1.w
        asgd2 = ASGD(obj, agrad,train_x, train_y, T = 10)
        asgd2.train()
        w2 = asgd2._w
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w1, 'wo')
        #ax = fig.add_subplot(212)
        ax.plot(w2, 'r.')
        plt.show()
        #for i in xrange(31):
            #if w1[i] != w2[i]:
                #a = 10
    r = asgd.predict1(test_x)
if __name__ == '__main__':
    #testSGD()
    #testAsgd()
    #fineDiff()
    testASGD()

    
    
    
