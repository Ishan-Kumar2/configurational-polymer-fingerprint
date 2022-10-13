#include "vars.h"
#include "utils.h"
#include "objects.h"
double Distance2(vector<double>& a, vector<double>& b, Input& in)//Distance^2 between two particles, any dimension - No periodic correction
{
    double dist=0.0;
    for(int i=0; i<in.NDIM; i++)
	dist+=(a[i]-b[i])*(a[i]-b[i]);
    return dist;
}

