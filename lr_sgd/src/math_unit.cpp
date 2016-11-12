/*************************************************************************
	> File Name: math_unit.cpp
	> Author: 
	> Mail: 
	> Created Time: 2015年09月03日 星期四 22时18分01秒
 ************************************************************************/

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include "math_unit.h"
using namespace std;

MathUnit::MathUnit(int n) : _N(n)
{
}
MathUnit::~MathUnit()
{
}
double MathUnit::Dot(vector<double>& v1, vector<double>& v2)
{
    double r = 0.0;
    for(vector<double>::iterator it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(), it2 != v2.end(); it1++, it2++)
    {
        r += (*it1) * (*it2);
    }
    return r;
}
double MathUnit::Dot(vector<int>& id, vector<double>& xi, vector<double>& w)
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
vector<double> MathUnit::Multiply(double val, vector<int>& id, vector<double>& v)
{
    vector<double> r(_N, 0);
    vector<int>::iterator id_it = id.begin();
    for(vector<double>::iterator it = v.begin(); it != v.end(); it++, id_it++)
    {
    	int i = *id_it;
    	double vi = *it;
        r[i] = val * vi;
    }
    return r;
}
vector<double> MathUnit::Multiply(double val, vector<double>& v)
{
    vector<double> r;
    for(vector<double>::iterator it = v.begin(); it != v.end(); it++)
    {
        r.push_back(val * (*it));
    }
    return r;
}
vector<double> MathUnit::Multiply(vector<double>& v1, vector<double>& v2)
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
    
vector<double> MathUnit::Subtract(vector<double>& v1, vector<double>& v2)
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

vector<double> MathUnit::Add(vector<double>& v1, vector<double>& v2)
{
    vector<double>::iterator it1;
    vector<double>::iterator it2;
    vector<double> r;

    for(it1 = v1.begin(), it2 = v2.begin(); it1 != v1.end(), it2 != v2.end(); it1++, it2++)
    {
    	double a = *it1;
    	double b = *it2;
        r.push_back(a + b);
    }
    return r;
}

int MathUnit::Sign(double v)
{
    if(v > 0.0)
        return 1;
    else if(v < 0.0)
        return -1;
    else
    	return 0;
}

vector<double> MathUnit::Sqrt(vector<double>& v)
{
    vector<double> r;
    for(vector<double>::iterator it = v.begin(); it != v.end(); it++)
    {
        r.push_back(sqrt(*it));
    }
    return r;
}

double MathUnit::Norm(vector<double>& v)
{
	double res = 0.0;
	for(vector<double>::iterator it = v.begin(); it != v.end(); it++)
	{
		res += (*it) * (*it);
	}
	return sqrt(res);
}
double MathUnit::Sum(vector<double>& v)
{
	double res = 0.0;
	for(vector<double>::iterator it = v.begin(); it != v.end(); it++)
	{
		res += (*it);
	}
	return res;
}






