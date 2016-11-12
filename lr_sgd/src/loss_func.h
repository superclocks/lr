/*************************************************************************
	> File Name: loss_func.h
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: 2015年05月28日 星期四 10时01分01秒
 ************************************************************************/
#ifndef _LOSS_FUNC_
#define _LOSS_FUNC_

#include<iostream>
using namespace std;
class LogLoss
{
	public:
		static double loss(double wTx, double y);
		static double dloss(double wTx, double y);
		static double deci(double z);
};


class LogLoss01
{
	public:
		static double loss(double wTx, double y);
		static double dloss(double wTx, double y);
		static double deci(double z);
};

class HingeLoss
{
	public:
		static double loss(double wTx, double y);
		static double dloss(double wTx, double y);
		static double deci(double z);
};


class SquaredHingeLoss
{
	public:
		static double loss(double wTx, double y);
		static double dloss(double wTx, double y);
		static double deci(double z);
};


class SooothHingeLoss
{
	public:
		static double loss(double wTx, double y);
		static double dloss(double wTx, double y);
		static double deci(double z);
};

#endif

