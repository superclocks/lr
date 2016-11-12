/*************************************************************************
	> File Name: sgd.cpp
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: 2015年05月27日 星期三 08时57分13秒
 ************************************************************************/
#include <fstream>
#include <iostream>
#include <iostream>
#include "svrg.h"
#include "ftrl.h"
int TestEigen();
using namespace std;

int main(int argv, char** argc)
{
	TestEigen();
    FTRL* ftrl = new FTRL(785, 1.0, 1.0, 0.1, 0.0,  1, LogLoss_);
    ftrl->Train("./mnist_train01.txt");
    ftrl->SaveModel("model.mldeo");
    ftrl->Evaluate("./mnist_test01.txt"); 
    
    //test(LogLoss01);
    /*readLine();
    exit(1);
	double* mg;
	char* train_path = argc[1];
	char* test_path = argc[2];

	int m = 700;
	int n = 30;
	int k = atoi(argc[3]);
	double rat = atof(argc[4]);
	double tol = 0.0001;
	double* w = (double*)malloc(sizeof(double) * n);
	svrgTrain(train_path, sigmoid, grad, m, n, k, rat, tol, w);
	int i;
	for(i = 0; i < n; i++)
	{
		printf("%lf\n", w[i]);
	}


	predictor(test_path, w, 30);
	free(w);*/
	return 0;
}
