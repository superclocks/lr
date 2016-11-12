#ifndef _MLR_
#define _MLR_
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "ftrl.h"


using namespace std;

class MLR : public FTRL
{
private:
		vector<double> _pi; //混合模型中，子模型前的系数
		vector<vector<double> > _gama; //EM算法中每个样本对应的系数
		vector<vector<double> > _ws; //混合模型
		int _K; //逻辑回归模型的个数
 public:
		MLR(int N, int K, double alfa = 1.0, double beta = 1.0, double lambda1 = 1.0,
        		double lambda2 =0.0, int iter = 50, double tol = 1e-5,
        		enum LossType loss_type = LogLoss_, enum BasisFunc basis_func = Default_);
		virtual ~MLR();
		virtual void Train(string path);
		double GamaElement(vector<int>& id, vector<double>& xi, vector<double>& w, int yi, double pi);
		void EStep(char* file_path);
		void MStep(string file_path);


};



#endif
