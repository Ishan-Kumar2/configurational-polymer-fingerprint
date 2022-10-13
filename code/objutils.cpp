#include "vars.h"
#include "utils.h"
#include "objects.h"
#include "objutils.h"
#include <cmath>
#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include<iostream>
using namespace Spectra;
double Elastic(Bead& b1, Bead& b2, Input& in)
{
   return 0.5*in.SPRING*Distance2(b1.coord,b2.coord,in); 
}

double DeltaEnergy(Bead& bn, vector<Bead>& b, Input& in)
{
    double temp=0.0;
    for(int i=0; i<int(bn.nbr.size()); i++)
      temp+=Elastic(bn,b[bn.nbr[i]],in)-Elastic(b[bn.index],b[bn.nbr[i]],in);
    return temp;
}

double TotalEnergy(vector<Bead>& b, Input& in)
{
    double temp=0.0;
    for(int i=0; i<in.NBEAD; i++)
    {
      for(int j=0; j<int(b[i].nbr.size()); j++)
	 if(b[i].nbr[j]>i)
	   temp+=Elastic(b[i],b[b[i].nbr[j]],in);
    }
    return temp;
}

double calcLJ(double dist)
{
	if(dist > pow(2, 1/6))
	{
	return 0.0;
	}
	else return 4.0*((1.0/pow(dist,6)) - (1.0/pow(dist,3)));
}

double LennardJones(vector<Bead>& b, Input& in)
{
    double temp=0.0;
    double d = 0.0;
    for(int i=0; i<in.NBEAD; i++)
    {
      for(int j=i+1; j<in.NBEAD; j++)
	 {
        if(i==j)
	       continue;
	 
	   d = Distance2(b[i].coord,b[j].coord,in);
	   temp+=calcLJ(d);
    	
      }
    }
    return temp;
}

double DeltaLennardJones(vector<Bead>& b, int index, Bead& bn, Input& in)
{
    long double t=0.0;
    long double d1 = 0.0;
    long double d2 = 0.0;
    for(int i=0; i<in.NBEAD; i++)
    {
      if(i!=index)
      {
      	d1 = Distance2(b[i].coord,b[index].coord,in);
	    t-=calcLJ(d1);
	    d2 = Distance2(b[i].coord,bn.coord,in);
	    t+=calcLJ(d2);	 
      }
  }
    return t;
}
double x,y,z;
vector<double> calcRg3D(vector<Bead>& pos, Input& in, double& r_cm_x, double& r_cm_y, double& r_cm_z)
{

	Eigen::MatrixXd rg_tensor {{0.0,0.0,0.0},{0.0, 0.0,0.0},{0.0,0.0,0.0}};
    
    int len = in.NBEAD;
    
    for(int i=0;i<len;i++)
    {
        x = pos[i].coord[0];
        y = pos[i].coord[1];
        z = pos[i].coord[2];
    	rg_tensor(0,0)+=(x - r_cm_x) * (x - r_cm_x);
    	rg_tensor(1,1)+=(y - r_cm_y) * (y - r_cm_y);
    	rg_tensor(2,2)+=(z - r_cm_z) * (z - r_cm_z);

    	rg_tensor(0,1)+=(x - r_cm_x) * (y - r_cm_y);
    	rg_tensor(0,2)+=(x - r_cm_x) * (z - r_cm_z);
    	rg_tensor(1,2)+=(y - r_cm_y) * (z - r_cm_z);
    }
    
    for(int i=0;i<3;i++)
    {
    	for(int j=0;j<3;j++)
    	{
    		rg_tensor(i,j)/=(len);
    	}
    }
    rg_tensor(1,0)=rg_tensor(0,1);
 	rg_tensor(2,0)=rg_tensor(0,2);
    rg_tensor(2,1)=rg_tensor(1,2);
    
    DenseSymMatProd<double> op(rg_tensor);
    SymEigsSolver< DenseSymMatProd<double> > eigs(op, 2,3);
    // Initialize and compute eigenvalues
    eigs.init();
    int nconv = eigs.compute(SortRule::LargestMagn);
    Eigen::VectorXd evalues;
    evalues = eigs.eigenvalues();
    //Sum of eigenvalues is equal to the trace of a matrix
    double trace = rg_tensor(0,0) + rg_tensor(1,1) + rg_tensor(2,2);

    vector<double> r {evalues(0,0), evalues(1,0), trace - (evalues(1,0) + evalues(0,0))};
    
    return r;
}

vector<double> sideLength(vector<Bead>& pos, Input& in)
{
    double max_x = -1000;
    double max_y = -1000;
    double max_z = -1000;
    double min_x = 1000;
    double min_y = 1000;
    double min_z = 1000;

    for(int i=0;i<in.NBEAD;i++)
    {
        max_x = max(max_x,pos[i].coord[0]);
        max_y = max(max_y,pos[i].coord[1]);
        max_z = max(max_z,pos[i].coord[2]);

        min_x = min(min_x,pos[i].coord[0]);
        min_y = min(min_y,pos[i].coord[1]);
        min_z = min(min_z,pos[i].coord[2]);
    }

    return {max_x-min_x, max_y-min_y, max_z-min_z};
}

double ShapeAnisotrpy(double& x, double& y, double& z)
{
    double dr = pow((pow(x,2) + pow(y,2) + pow(z,2)), 2);
    double nr = (pow(x,4) + pow(y,4) + pow(z,4));
    double sa = ((3* nr)/(2*dr)) - (1/2);
    return sa;
}
