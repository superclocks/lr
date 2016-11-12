
#ifndef _MATH_UNIT_
#define _MATH_UNIT_

#include <vector>
#include <math.h>
using namespace std;

class MathUnit
{
	private:
		int _N;
    public:
		MathUnit(int n);
		virtual ~MathUnit();
		double Sum(vector<double>& v);
        double Dot(vector<int>& id, vector<double>& vals, vector<double>& w);
        double Dot(vector<double>& v1, vector<double>& v2);
        vector<double> Multiply(double val, vector<double>& v);
        vector<double> Multiply(double val, vector<int>& id , vector<double>& v);
        vector<double> Multiply(vector<double>& v1, vector<double>& v2);
        vector<double> Add(vector<double>& v1, vector<double>& v2);
        vector<double> Subtract(vector<double>& v1, vector<double>& v2);
        vector<double> Sqrt(vector<double>& v);
        double Norm(vector<double>& v);
        int Sign(double v);
        /*void SaveModel(char* path);
        double Predict(vector<int>& id, vector<double>& x);
        void Evaluate(char* f);*/
};

#endif
