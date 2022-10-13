#ifndef _VARS_H
#define _VARS_H
#include<iostream>
#include<iomanip>
#include<cmath>
#include<string>
#include<vector>
#include<cstdio>
#include<fstream>
#include<ctime>
#include<gsl/gsl_rng.h>
#include<boost/program_options.hpp>
#include<time.h>
using namespace std;
using namespace boost::program_options;

//Default Params
//3, 1000, 1000, 10, 0.75

class Input
{
  public:
    int NDIM,SWEEP,SAMPLE,NBEAD;
    double MCSTEP, ENERGY,SPRING;
    string SAVE_LOC = "./Dataset";
    const gsl_rng_type * gsl_T;
    gsl_rng * gsl_r;
    Input(){
      NDIM=3; SWEEP=1000; SAMPLE=10; NBEAD=1000; MCSTEP=7.5; ENERGY=0.0; SPRING=3;
      gsl_rng_env_setup();
      gsl_T = gsl_rng_default;
      gsl_r = gsl_rng_alloc (gsl_T);
      gsl_rng_set(gsl_r, time(NULL)*time(NULL));
    }
    ~Input(){
      gsl_rng_free(gsl_r);
    }
    void ReadInput(int argc, char* argv[])
    {
	  options_description desc("Usage:\nMC <options>");
	  desc.add_options()
	  ("help,h", "print usage message")
	  ("NDIM,d", value<int>(&NDIM)->default_value(3), "No. of dimensions (defult 3)")
	  ("NBEAD,n", value<int>(&NBEAD)->default_value(100), "number of beads in the system (default 1000)")
	  ("SWEEP,s", value<int>(&SWEEP)->default_value(10000000), "#MC sweeps (default 1000)")
	  ("SAMPLE,S", value<int>(&SAMPLE)->default_value(100000), "Sampling Frequency (default 10")
	  ("MCSTEP,m", value<double>(&MCSTEP)->default_value(3.85), "MC step (default 0.75)")
	  ("SAVE_LOC,f", value<string>(&SAVE_LOC)->default_value("./Dataset"), "Save Location");

	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);
	notify(vm);

	if (vm.count("help"))
	{
	  cout << desc << "\n";
	  exit(1);
	}
	//Set spring constant to NDIM
	SPRING= (1 + gsl_rng_uniform(gsl_r) * 6);
    }
};
#endif
