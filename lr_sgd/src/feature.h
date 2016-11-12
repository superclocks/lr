/*************************************************************************
	> File Name: feature.h
	> Author: 
	> Mail: 
	> Created Time: 2015年09月27日 星期日 22时59分21秒
 ************************************************************************/

#ifndef _FEATURE_
#define _FEATURE_
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
using namespace std;
using namespace boost;
class Feature
{

    public:
       static void Default(string& s, vector<int>& _id, vector<double>& _xi, double& _yi);
};

#endif
