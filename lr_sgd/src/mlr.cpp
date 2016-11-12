#include "mlr.h"
#include "ftrl.h"
#include "def.h"
#include "feature.h"
MLR::MLR(int N, int K, double alfa, double beta, double lambda1,
		double lambda2, int iter, double tol,
		enum LossType loss_type, enum BasisFunc basis_func) :
		FTRL(N, alfa, beta, lambda1, lambda2, iter, tol, loss_type, basis_func)
{
		this->_K = K; //模型的个数
		for(int i = 0; i < _K; i++)
		{
				vector<double> w(_N, 0.0);
				_ws.push_back(w);
				_pi.push_back(0.2);
		}
}
double MLR::GamaElement(vector<int>& _id, vector<double>& _xi, vector<double>& _w, int yi, double pi)
{
		double wTx = Dot(_id, _xi, _w);
		double loss_val = 0.0;
		if(_loss_type == LogLoss_)
				loss_val = LogLoss::dloss(wTx, _yi);
		else if(_loss_type == LogLoss01_)
				loss_val = LogLoss01::dloss(wTx, _yi);
		double r = pi * pow(loss_val, yi) * pow(1.0 - loss_val, 1.0 - yi);
		return r;

}
void MLR::EStep(char* file_path)
{
		string line;
		//按照每行读取数据
		vector<double> xi;
		vector<int> id;
		double yi;
		ifstream reader(file_path, ios::in);
		while(getline(reader, line))
		{
					Feature::Default(line, id, xi, yi);
					vector<double> tmp;
					double num = 0.0;
					for(int i = 0; i < _K; i++)
					{
							double pi = _pi[i];
							vector<double> wi = _ws[i];
							double r = GamaElement(_id, _xi, wi, _yi, pi);
							tmp.push_back(r);
							num += r;
					}
					for(int i = 0; i < _K; i++)
					{
							tmp[i] = tmp[i] / num;
					}
					_gama.push_back(tmp);
		}
		reader.close();

}
void MLR::Train(string file_path)
{

}
MLR::~MLR()
{

}
