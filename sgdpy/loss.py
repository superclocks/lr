import numpy as np

class LogLoss:
    #logloss(a,y) = log(1+exp(-a*y))
    def loss(self, wTx, y):
        z = wTx * y
        if z > 18:
            return np.exp(-z)
        if z < -18:
            return -z
        return np.log2(1.0 + np.exp(-z))
    #-dloss(a,y)/da
    def dloss(self, wTx, y):
        z = wTx * y
        if z > 18:
            return -y * np.exp(-z)
        if z < -18:
            return -y
        
        return -float(y) / (1.0 + np.exp(z))
        
    def deci(self, w, x):
        z = w.dot(x)
        print z
        return 1.0 / (1 + np.exp(-z))
class LogLoss01:
    #logloss(a,y) = - (y*log(h(x)) + (1 - y) * log(1 - h(x)))
    #h(x) = 1 / (1 + exp(-a * x))
    def loss(self, wTx, y):
        if wTx > 18:
            h = np.exp(-wTx)
        elif wTx < -18:
            h = 1.0 / (1.0 - wTx) 
        else:
            h = 1.0 / (1.0 + np.exp(-wTx))        
        return -(y * np.log2(h) + (1.0 - y) * np.log2(1.0 - h))
    #-dloss(a,y)/da
    def dloss(self, wTx, y):
        if wTx > 18:
            h = np.exp(-wTx)
        elif wTx < -18:
            h = 1.0 / (1.0 - wTx) 
        else:
            h = 1.0 / (1.0 + np.exp(-wTx))  
        return h - y
        
    def deci(self, w, x):
        z = w.dot(x)
        return 1.0 / (1 + np.exp(-z))
class HingeLoss:
    #hingeloss(a,y) = max(0, 1-a*y)
    def loss(self, wTx, y):
        z = wTx * y
        if z > 1:
            return 0
        return 1 - z
    #-dloss(a,y)/da
    def dloss(self, wTx, y):
        z = wTx * y
        if z > 1:
            return 0
        return -y
    
    def deci(self, w, x):
        z = w.dot(x)
        return z
    
class SquaredHingeLoss:
    #squaredhingeloss(a,y) = 1/2 * max(0, 1-a*y)^2
    def loss(self, wTx, y):
        z = wTx * y
        if z > 1:
            return 0
        d = 1.0 - z
        return 0.5 * d * d
    
    def dloss(self, wTx, y):
        z = wTx * y
        if z > 1:
            return 0
        return -y * (1.0 - z)
    
    def deci(self, w, x):
        z = w.dot(x)
        return z
    
class SmoothHingeLoss:
    def loss(self, wTx, y):
        z = wTx * y
        if (z > 1):
            return 0;
        if (z < 0):
            return 0.5 - z;
        d = 1 - z;
        return 0.5 * d * d;

    #-dloss(a,y)/da
    def dloss(self, wTx, y):
        z = wTx * y;
        if (z > 1): 
            return 0;
        if (z < 0):
            return -y;
        return -y * (1 - z);
    
    def deci(self, w, x):
        z = w.dot(x)
        return z
  