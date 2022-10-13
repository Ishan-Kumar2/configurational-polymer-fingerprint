#include "vars.h"
#include "utils.h"
#include "objects.h"
#include "montecarlo.h"
void Initialize(vector<Bead>&, Input& );
int main(int argc, char* argv[])
{
  srand(time(0));

    Input in;
    in.ReadInput(argc,argv);
    vector<Bead> b(in.NBEAD);
    
    //Initialize Beads
    Initialize(b,in);
    
    //MC Simulation
    MonteCarlo(b,in);  
    
}

void Initialize(vector<Bead>& b, Input& in)
{
    int i,j;
    b[0].index=0;
    b[0].nbr.push_back(1);
    for(j=0; j<in.NDIM; j++)
      b[0].coord.push_back(0.0);
    for(i=1; i<in.NBEAD-1; i++)
    {
      b[i].index=i;
      b[i].nbr.push_back(i+1);
      b[i].nbr.push_back(i-1);
      for(j=0; j<in.NDIM; j++)
	b[i].coord.push_back(b[i-1].coord[j]+(gsl_rng_uniform(in.gsl_r)-0.5));
    }
    
    b[in.NBEAD-1].index=in.NBEAD-1;
    b[in.NBEAD-1].nbr.push_back(in.NBEAD-2);
    for(j=0; j<in.NDIM; j++)
	b[in.NBEAD-1].coord.push_back(b[in.NBEAD-2].coord[j]+(gsl_rng_uniform(in.gsl_r)-0.5));
}
