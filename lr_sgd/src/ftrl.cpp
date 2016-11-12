/*************************************************************************
	> File Name: ftrl.cpp
	> Author: superclocks@163.com
	> Mail:
	> Created Time: 2015年09月03日 星期四 22时18分01秒
 ************************************************************************/

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "ftrl.h"
#include "feature.h"

using namespace std;
using namespace boost;

FTRL::FTRL(int N, double alfa, double beta, double lambda1, double lambda2,
            int iter, double tol, enum LossType loss_type, enum BasisFunc basis_func): _alfa(alfa), _beta(beta),
            _lambda1(lambda1), _lambda2(lambda2), _N(N), _iter(iter), _tol(tol), _loss_type(loss_type), _basis_func(basis_func)
{
	if(_basis_func == Default_)
	{
		_N = _N + 1;
	}
	vector<double> w(_N, 0.0);
    _w = w;

}

FTRL::~FTRL()
{

}

void FTRL::ParseLine(string s)
{
    vector<string> ele;
    split(ele, s, is_any_of(" "));
    vector<string>::iterator it = ele.begin();
    _yi = lexical_cast<double>(ele[0]);
    it++;
    for(; it != ele.end(); it++)
    {
        vector<string> tmp;
        split(tmp, *it, is_any_of(":"));
        int id = lexical_cast<int>(tmp[0]);
        double val =  lexical_cast<double>(tmp[1]);
        _id.push_back(id);
        _xi.push_back(val);
    }
}
double FTRL::Dot(vector<double>& v1, vector<double>& v2)
{
    double r;
    for(vector<double>::iterator it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(), it2 != v2.end(); it1++, it2++)
    {
        r += (*it1) * (*it2);
    }
    return r;
}
double FTRL::Dot(vector<int>& id, vector<double>& xi, vector<double>& w)
{
    double v = 0.0;
    int index;
    for(size_t i = 0; i < id.size(); i++)
    {
        index = id[i];
        double t1 = w[index];
        double t2 = xi[i];
        v += t1 * t2;
    }
    return v;
}
vector<double> FTRL::Multiply(double val, vector<int>& id, vector<double>& v)
{
    vector<double> r(_N, 0.0);
    vector<int>::iterator id_it = id.begin();
    for(vector<double>::iterator it = v.begin(); it != v.end(); it++, id_it++)
    {
    	int i = *id_it;
    	double vi = *it;
        r[i] = val * vi;
    }
    return r;
}
vector<double> FTRL::Multiply(double val, vector<double>& v)
{
    vector<double> r;
    for(vector<double>::iterator it = v.begin(); it != v.end(); it++)
    {
        r.push_back(val * (*it));
    }
    return r;
}
vector<double> FTRL::Multiply(vector<double>& v1, vector<double>& v2)
{
    vector<double>::iterator it1;
    vector<double>::iterator it2;
    vector<double> r;
    for(it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(), it2 != v2.end(); it1++, it2++)
    {
        r.push_back((*it1) * (*it2));
    }
    return r;
}

vector<double> FTRL::Subtract(vector<double>& v1, vector<double>& v2)
{

    vector<double>::iterator it1;
    vector<double>::iterator it2;
    vector<double> r;

    for(it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(), it2 != v2.end(); it1++, it2++)
    {
    	double t1 = *it1;
    	double t2 = *it2;
        r.push_back(t1 - t2);
    }
    return r;
}

vector<double> FTRL::Add(vector<double>& v1, vector<double>& v2)
{
    vector<double>::iterator it1;
    vector<double>::iterator it2;
    vector<double> r;

    for(it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(), it2 != v2.end(); it1++, it2++)
    {
        r.push_back((*it1) + (*it2));
    }
    return r;
}

int FTRL::Sign(double v)
{
    if(v > 0.0)
        return 1;
    else if(v < 0.0)
        return -1;
    else
    	return 0;
}
void FTRL::UpdateW(vector<double>& z, vector<double>& q)
{
    for(int i = 0; i < _N; i++)
    {
        if(abs(z[i]) < _lambda1)
            _w[i] = 0.0;
        else
        {
            _w[i] = -1.0 / (_lambda2 + (_beta + sqrt(q[i])) / _alfa) * (z[i] - _lambda1 * Sign(z[i]));
        }
    }
}
vector<double> FTRL::Sqrt(vector<double>& v)
{
    vector<double> r;
    for(vector<double>::iterator it = v.begin(); it != v.end(); it++)
    {
        r.push_back(sqrt(*it));
    }
    return r;
}

void FTRL::SaveModel(char* path)
{
	ofstream writer(path, ios::out);
	int id = 0;
	for(vector<double>::iterator it = _w.begin(); it != _w.end(); it++)
	{
		writer << id++ << "\t" << *it << endl;
	}
	writer.close();
}
double FTRL::Norm(vector<double>& v)
{
	double res = 0.0;
	for(vector<double>::iterator it = v.begin(); it != v.end(); it++)
	{
		res += (*it) * (*it);
	}
	return sqrt(res);
}
void FTRL::Train(char* file_path)
{
    string line;
    //按照每行读取数据
    vector<double> q(_N, 0.0);
    vector<double> z(_N, 0.0);
    int iter = 0;
    vector<double> old_w(_N, 0.0);

    while(iter++ < _iter)
    {
    	ifstream reader(file_path, ios::in);
		while(getline(reader, line))
		{
			//cout << line << endl;
			//ParseLine(line); //将每行libsvm数据格式转换为数值格式
			Feature::Default(line, _id, _xi, _yi);
            double wTx = Dot(_id, _xi, _w);
			double loss_val = 0.0;
			if(_loss_type == LogLoss_)
				loss_val = LogLoss::dloss(wTx, _yi);
			else if(_loss_type == LogLoss01_)
				loss_val = LogLoss01::dloss(wTx, _yi);
			vector<double> g = Multiply(loss_val, _id,  _xi);
			vector<double> gg = Multiply(g, g);
			vector<double> cul_q = Add(q, gg);
			vector<double> t0 = Sqrt(q);
			vector<double> t = Multiply(1.0 / _alfa, t0);
			vector<double> t1 = Sqrt(cul_q);
			vector<double> sigma = Subtract(t1, t);
			q = Add(q, gg);
			vector<double> e0 = Multiply(sigma, _w);
			vector<double> e1 = Subtract(g, e0);
			z = Add(z, e1);
			//updata _w
			UpdateW(z, q);
			_xi.clear();
			_id.clear();
		}
		reader.close();

		vector<double> del_w = Subtract(_w, old_w);
		if(Norm(del_w) <= _tol)
			break;
		old_w = _w;
    }
}

double FTRL::Predict(vector<int>& id, vector<double>& x)
{
    if(_loss_type == LogLoss_)
    {
        double z = Dot(id, x, _w);
        double pi = LogLoss::deci(z);
        return pi;
        /*if(pi > 0.5)
            return 1;
        else
            return 0;*/
    }
    else if(_loss_type == LogLoss01_)
    {
        double z = Dot(id, x, _w);
        double pi = LogLoss01::deci(z);
        return pi;
        /*if(pi > 0.5)
            return 1;
        else
            return -1;*/
    }
}

void FTRL::Evaluate(char* f)
{
    ifstream reader(f, ios::in);
    if(!reader.is_open())
    {
        cout<< "Open file failed." << endl;
        assert(-1);
    }
    string s;
    vector<int> ids;
    vector<double> x;
    int k = 0;
    int count = 0;
    while(getline(reader, s))
    {

        /*vector<string> ele;
        split(ele, s, is_any_of(" "));
        vector<string>::iterator it = ele.begin();
        int y = lexical_cast<int>(ele[0]);
        it++;
    	x.clear();
    	ids.clear();
        for(; it != ele.end(); it++)
        {
            vector<string> tmp;
            split(tmp, *it, is_any_of(":"));
            int id = lexical_cast<int>(tmp[0]);
            double val =  lexical_cast<double>(tmp[1]);
            ids.push_back(id);
            x.push_back(val);
        }*/
        double y;
        ids.clear();
        x.clear();
        Feature::Default(s, ids, x, y);
        int y_pred;
        double pi = Predict(ids, x);
        if(_loss_type == LogLoss_)
            y_pred = pi > 0.5 ? 1 : -1;
        else if(_loss_type == LogLoss01_)
            y_pred = pi > 0.5 ? 1 : 0;
//cout<<  "pi = " << pi <<  "y_pred = " << y_pred << " y = " << y << endl;
        if(y_pred != y)
            k++;
        count++;
    }
    cout << "error rating = " << (double)k / count << endl;
}
