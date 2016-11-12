/*************************************************************************
	> File Name: Feature.cpp
	> Author: 
	> Mail: 
	> Created Time: 2015年09月27日 星期日 23时02分26秒
 ************************************************************************/
#include "feature.h"

void Feature::Default(string& s, vector<int>& _id, vector<double>& _xi, double& _yi)
{
    vector<string> ele;
    split(ele, s, is_any_of(" "));
    vector<string>::iterator it = ele.begin();
    _yi = lexical_cast<double>(ele[0]);
    _id.push_back(0);
    _xi.push_back(1.0);
    
    it++;
    for(; it != ele.end(); it++)
    {
        vector<string> tmp;
        split(tmp, *it, is_any_of(":"));
        int id = lexical_cast<int>(tmp[0]);
        double val =  lexical_cast<double>(tmp[1]);
        _id.push_back(id + 1);
        _xi.push_back(val);
    }    
}

