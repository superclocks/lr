/*************************************************************************
	> File Name: loss_func.cpp
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: 2015年05月28日 星期四 10时03分13秒
 ************************************************************************/

#include <iostream>
#include <math.h>
#include "loss_func.h"
using namespace std;
//LogLoss
double LogLoss::loss(double wTx, double y)
{
	double z = wTx * y;
	if(z > 18.0)
		return exp(-z);
	else if(z < -18.0)
		return -z;
	return log2(1.0 + exp(-z));
}
double LogLoss::dloss(double wTx, double y)
{
	double z = wTx * y;
	if(z > 18.0)
		return -y * exp(-z);
	else if(z < -18)
		return -y;
	return -y / (1.0 + exp(z));
}
double LogLoss::deci(double z)
{
	return 1.0 / (1.0 + exp(-z));
}

//LogLoss01
double LogLoss01::loss(double wTx, double y)
{
	double h;
	if(wTx > 18.0)
		h = exp(-wTx);
	else if(wTx < -18.0)
		h = 1.0 /(1.0 - wTx);
	else
		h = 1.0 / (1.0 + exp(-wTx));
	return -(y * log2(h) + (1.0 - y) *log2(1.0 - h));
}
double LogLoss01::dloss(double wTx, double y)
{
	double h;
	if(wTx > 18.0)
		h = exp(-wTx);
	else if(wTx < -18)
		h = 1.0 / (1.0 - wTx);
	else
		h = 1.0 / (1.0 + exp(-wTx));
	return h - y;
}
double LogLoss01::deci(double z)
{
	return 1.0 / (1.0 + exp(-z));
}





