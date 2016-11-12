/*************************************************************************
	> File Name: sgd.h
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: 2015年05月27日 星期三 08时57分13秒
 ************************************************************************/
#ifndef _SVRG_
#define _SVRG_

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "feature.h"
#include "loss_func.h"
#include "def.h"

using namespace std;



class SVRG : public Feature
{
	public:
		int _m;
		int _k;
		double _lambda;
		int _ita;
		vector<int> _id; //每行数据的特征id
		vector<double> _xi; //每行数据的特征值
		vector<double> _w; //模型参数
		double _yi; //每行数据的y值
		enum LossType _loss_type;
	    enum BasisFunc _basis_func;
	public:
		 void Train(string file_path);
		 vector<double> gradMean(vector<double>& g);
};

#endif
