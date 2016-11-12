/*************************************************************************
	> File Name: ftrl.h
	> Author: 
	> Mail: 
	> Created Time: 2015年09月03日 星期四 22时17分07秒
 ************************************************************************/

#ifndef _FTRL_
#define _FTRL_

#include <vector>
#include <math.h>
#include "def.h"
#include "loss_func.h"
using namespace std;

class FTRL
{
    public:
		//double (*_dloss)(double wTx, double y);
        double _alfa;
        double _beta;
        double _lambda1;
        double _lambda2;
        int _N; //特征的个数
        int _iter; //迭代次数
        double _tol;
        vector<int> _id; //每行数据的特征id
        vector<double> _xi; //每行数据的特征值
        vector<double> _w; //模型参数
        double _yi; //每行数据的y值
        //dloss1 _dloss; //损失函数导数函数指针
        enum LossType _loss_type;
        enum BasisFunc _basis_func;
    public:
        FTRL(int N, double alfa = 1.0, double beta = 1.0, double lambda1 = 1.0,
        		double lambda2 =0.0, int iter = 50, double tol = 1e-5,
        		enum LossType loss_type = LogLoss_, enum BasisFunc basis_func = Default_);
       virtual  ~FTRL();
        void ParseLine(string s);
        virtual void Train(char* file_path);
        double Dot(vector<int>& id, vector<double>& vals, vector<double>& w);
        double Dot(vector<double>& v1, vector<double>& v2);
        vector<double> Multiply(double val, vector<double>& v);
        vector<double> Multiply(double val, vector<int>& id , vector<double>& v);
        vector<double> Multiply(vector<double>& v1, vector<double>& v2);
        vector<double> Add(vector<double>& v1, vector<double>& v2);
        vector<double> Subtract(vector<double>& v1, vector<double>& v2);
        vector<double> Sqrt(vector<double>& v);
        void UpdateW(vector<double>& z, vector<double>& q);
        double Norm(vector<double>& v);
        int Sign(double v);
        void SaveModel(char* path);
        double Predict(vector<int>& id, vector<double>& x);
        void Evaluate(char* f);
};

#endif
