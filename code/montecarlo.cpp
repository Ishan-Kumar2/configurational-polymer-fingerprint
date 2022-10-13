#include "vars.h"
#include "objects.h"
#include "utils.h"
#include "objutils.h"
#include "montecarlo.h"
#include<string>
#include<sys/time.h>
#include<cmath>
#define pi 3.14159

double max_move = 4;
double r_cm_x = 0.0;
double r_cm_y = 0.0;
double r_cm_z = 0.0;
double theta = 0.0;
double phi = 0.0;
double r = 0.0;
double max_u = 0.0;
double min_u = 1000000;
float _lower_limit = 0.7;
float _upper_limit = 1.3;
bool do_bad=true;
double temp = 0.0;
double step_bad_move;
vector<Bead> temp2;

void bad_moves(vector<Bead>& b, int i, Input& in, bool mul=true)
{
	ofstream out;
	//Copy everything
	double energy = in.ENERGY;
	vector<Bead> pos = b;
	Bead bn;
	
    for(int move=0;move<int(max_move);move++)
    {
    	double index=gsl_rng_uniform_int(in.gsl_r,in.NBEAD);
    	bn=pos[index];
    	theta = 2 * pi * ((double)rand()/(RAND_MAX));
		phi = acos(2 * ((double)rand()/(RAND_MAX))-1);
		if(mul)
		r=((double)rand()/(RAND_MAX))*step_bad_move*1.90;
		else
		r=((double)rand()/(RAND_MAX))*step_bad_move;
			
		bn.coord[0] += r * sin(phi) * cos(theta);
		bn.coord[1] += r * sin(phi) * sin(theta);
		bn.coord[2] += r * cos(phi);
		double delta=DeltaEnergy(bn,pos,in);// + DeltaLennardJones(b,index,bn,in);
		//Accept
		energy+=delta;
		pos[index] = bn;
    }
    
   	if(energy>750)
   	{
   		if(max_move>6)
   			max_move-=1;
   		// cout<<"TOO HIGH"<<endl;
   		return; 
   	}	

   	if(energy<=170)
   	{
   		// cout<<"TOO LOW "<<energy<<endl;
   		return;
   	}
   		
	double d = Distance2(pos[0].coord, pos[in.NBEAD-1].coord, in);
	double d2 = calcRg(pos, in.NBEAD);
	vector<double> d3, d5;
    long double d4 = 0.0;

  	d3 = calcRg3D(pos, in, r_cm_x,r_cm_y,r_cm_z);
	d4 = ShapeAnisotrpy(d3[0],d3[1],d3[2]);
  	d5 = sideLength(pos, in);	    
    

  	if(energy >= max_u)
		max_u = energy;
	if(energy <= min_u)
		min_u = energy;

    string f2 = in.SAVE_LOC+ "B_" + to_string(i+1)+".txt";
    out.open(f2,ios::app);
    out<<"RE,"<<d;
    out<<"\nRG,"<<d2;
    out<<"\nInternal Energy,"<<energy;
    out<<"\nRG 3D,"<<d3[0];
    out<<"\nRG 3D,"<<d3[1];
    out<<"\nRG 3D,"<<d3[2];
    out<<"\nShape Anisotropy,"<<d4;
    out<<"\nSideX,"<<d5[0];
    out<<"\nSideY,"<<d5[1];
    out<<"\nSideZ,"<<d5[2];
	out<<"\nSpringConstant,"<<in.SPRING;
    for(int i =0;i<in.NBEAD;i++)
	{	
		out<<"\n"<<pos[i].coord[0]<<","<<pos[i].coord[1]<<","<<pos[i].coord[2];
	}    
    out.close();
    // cout<<"Energy Bad "<<energy<<"   "<<"Actual Energy"<<in.ENERGY<<"  "<<"Max move"<<max_move<<endl;

    if(i>in.SAMPLE && max_move<12)
    	max_move*=1.04;
	return;
}

double calcRg(vector<Bead>& pos, int len)
{
    
    for(int i=0;i<len;i++)
    {
    	r_cm_x+=pos[i].coord[0];
    	r_cm_y+=pos[i].coord[1];
    	r_cm_z+=pos[i].coord[2];
    }
    r_cm_x = r_cm_x/len;
    r_cm_y = r_cm_y/len;
    r_cm_z = r_cm_z/len;
    
    double r_x = 0.0;
    double r_y = 0.0;
    double r_z = 0.0;
    
    for(int i=0;i<len;i++)
    {
    	r_x+=(pos[i].coord[0] - r_cm_x) * (pos[i].coord[0] - r_cm_x);
    	r_y+=(pos[i].coord[1] - r_cm_y) * (pos[i].coord[1] - r_cm_y);
    	r_z+=(pos[i].coord[2] - r_cm_z) * (pos[i].coord[2] - r_cm_z);
    }
    
    return (r_x+r_y+r_z)/len;
}


int MCMove(vector<Bead>& b, int index, Input& in)//Single Bead Displacement: Pass all particles and index of particle to move, Return accept (1) or reject (0)
{
    //To store new position
    Bead bn;
    bn=b[index];
    
    ///////////Attempt a displacement////
    theta = 2 * pi * ((double)rand()/(RAND_MAX));
	phi = acos(2 * ((double)rand()/(RAND_MAX))-1);
	
	r=((double)rand()/(RAND_MAX))*in.MCSTEP;
    
	// r=in.MCSTEP;//* ((_upper_limit - _lower_limit) * ((float)rand() / RAND_MAX)) + _lower_limit;
	
	bn.coord[0] += r * sin(phi) * cos(theta);
	bn.coord[1] += r * sin(phi) * sin(theta);
	bn.coord[2] += r * cos(phi);

    ////////////////////////////////////
    // for(int i=0; i<in.NDIM; i++)
    // bn.coord[i]+=(gsl_rng_uniform(in.gsl_r)-0.5)*in.MCSTEP;
  	
    //Compute change in elastic energy
    double delta=DeltaEnergy(bn,b,in);// + DeltaLennardJones(b,index,bn,in);

    //Accept or reject the move
    if(exp(-delta)>gsl_rng_uniform(in.gsl_r))
    {
    // cout<<r<<endl;
      in.ENERGY+=delta;
      b[index]=bn;
      return 1;
    }
    else
      return 0;
}

void MonteCarlo(vector<Bead>& b,Input& in)
{
      time_t start = time ( NULL );
      in.ENERGY=TotalEnergy(b,in);// + LennardJones(b,in);
      //cout<<"Monte Carlo Started, Energy="<<in.ENERGY<<endl<<"\tDoing "<<in.SWEEP<<" sweeps"<<endl;
      //To string Output
      string filename;
      //sprintf (filename, "_MC_D_%d_N_%d.dat", in.NDIM, in.NBEAD);
      // sprintf (filename, "./outputs/RE%d.dat",  in.NBEAD);
      
      char filename_2[100];
      //sprintf (filename, "_MC_D_%d_N_%d.dat", in.NDIM, in.NBEAD);
      sprintf (filename_2, "./outputs/RG%d.dat",  in.NBEAD);
      
      
      ofstream out;
      //out.open(filename);
      //out<<"MC equilibration started on\t"<<ctime ( &start ) <<endl;
      //out<<"\nSweep\tEnergy\tAcceptance\tRe"<<endl;
      //out.close();
      
      int Acceptance=0,index=0;
      long double Re=0.0; //To store average R^2
      long double Rg=0.0;
      long double d = 0.0;
      long double d2 = 0.0;
      long double running_u = 0.0;
      double acceptance_sum = 0.0;
      string f2;
      step_bad_move = in.MCSTEP;

      vector<double> d3, d5;
      long double d4 = 0.0;

      //Equilibrium Moves
      for(int i=0; i<in.SWEEP; i++)
      {
      	  d = Distance2(b[0].coord,b[in.NBEAD-1].coord,in);
		  Re+=d;
		  d2 = calcRg(b, in.NBEAD); 
		  Rg+=d2;
		  running_u += in.ENERGY;
	  
	  if(i>(3*in.SAMPLE))
	  {
	  	  if(in.ENERGY>=max_u)
		  	max_u = in.ENERGY;
		  if(in.ENERGY<=min_u)
		  	min_u = in.ENERGY;
	  }
	  //Sample and print if needed
	  
	  /*
	  if(int(i/in.SAMPLE)==0)
	  {
		  if(i%(in.SAMPLE/100)==0)
		   {
			  string f2;
			  f2 = in.SAVE_LOC+ "Nu_" + to_string(i)+".txt";
			  out.open(f2,ios::app);
			  out<<d<<endl;
			  out<<d2<<endl;    
			    out.close();
			}	
	  }*/

	  if(i<in.SAMPLE && i%(in.SAMPLE/50)==0)
	  {
	  		bad_moves(b, i+3, in, false);
	  }

	  if(i%in.SAMPLE==0)
	  {   
	  	d3 = calcRg3D(b, in, r_cm_x,r_cm_y,r_cm_z);
		d4 = ShapeAnisotrpy(d3[0],d3[1],d3[2]);
	  	d5 = sideLength(b, in);	    
	
	    //out.open(filename,ios::app);
	    //out<<setw(8)<<i<<"\t"<<setw(8)<<in.ENERGY<<"\t"<<setw(8)<<double(Acceptance)/in.SAMPLE<<"\t"<<setw(8)<<Re/in.SAMPLE<<endl;
	    //out<<Re/in.SAMPLE<<endl;
	    //out.close();
	    
	    
	    //out.open(filename_2,ios::app);
	    //out<<Rg/in.SAMPLE<<endl;
	    //out.close();
	    
	    //Print whatever;
	    
	    //Uncomment this 
		// gettimeofday(&start, NULL);		
	    
	    f2 = in.SAVE_LOC+ "N_" + to_string(i)+".txt";
	    //cout<<"saving in "<<f2<<endl;
	    out.open(f2,ios::app);
	    // out<<d<<endl<<d2<<endl<<in.ENERGY<<endl<<d3[0]<<endl<<d3[1]<<endl<<d3[2]<<endl<<d4<<endl<<d5[0]<<endl<<d5[1]<<endl<<d5[2];
	    out<<"RE,"<<d;
	    out<<"\nRG,"<<d2;
	    out<<"\nInternal Energy,"<<in.ENERGY;
	    out<<"\nRG 3D,"<<d3[0];
	    out<<"\nRG 3D,"<<d3[1];
	    out<<"\nRG 3D,"<<d3[2];
	    out<<"\nShape Anisotropy,"<<d4;
	    out<<"\nSideX,"<<d5[0];
	    out<<"\nSideY,"<<d5[1];
	    out<<"\nSideZ,"<<d5[2];
	 //    out<<"\nRunning U,"<<running_u/in.SAMPLE;
		// out<<"\nRunning re,"<<Re/in.SAMPLE;
	 //    out<<"\nRunning rg,"<<Rg/in.SAMPLE;
		out<<"\nSpringConstant,"<<in.SPRING;
	    for(int i =0;i<in.NBEAD;i++)
	    {	
		 out<<"\n"<<b[i].coord[0]<<","<<b[i].coord[1]<<","<<b[i].coord[2];
	    }
	    out.close();
	    
		acceptance_sum +=Acceptance;
		if(i>=5000000 && i%400000 == 0)
		{
			
			if(acceptance_sum >= 4*68000)
	    		in.MCSTEP = in.MCSTEP * 1.1;
	    	else if(acceptance_sum < 4*68000)
	    		in.MCSTEP = in.MCSTEP * 0.98;
	    	acceptance_sum = 0;
		}
		if(i<5000000 && i%400000 == 0)
		{
			in.MCSTEP = in.MCSTEP*0.99;
		}
		
	    // cout<<"Acc "<<Acceptance<<"   "<<" Step_Size "<<in.MCSTEP<<" r "<<r<<" "<<acceptance_sum<<" "<<"U"<<in.ENERGY<<" "<<max_u<<" "<<min_u<<endl;
	    Acceptance=0;
	    Re=0.0;
	    Rg=0.0;
	    running_u = 0.0;

	    temp = in.ENERGY;
	    if(do_bad)
	    {	
	    	bad_moves(b, i, in);
	    }
	    assert(in.ENERGY == temp);
	    
		// do_bad = !do_bad;
	  }
	  // gettimeofday(&start, NULL);		
	    
	  //Choose a particle to move
	  index=gsl_rng_uniform_int(in.gsl_r,in.NBEAD);
	  //Move it
	  Acceptance+=MCMove(b, index,in);    
	  // cout<<index<<endl;
	  }


      start=time ( NULL );
      //out.open(filename,ios::app);
      //out<<"\nMC production started on\t"<<ctime ( &start )<<endl<<endl;
      //out.close();
      //Change sweep and sample if needed
      //... 
      Re=0.0;
      Rg=0.0;
      d = 0.0;
      d2 = 0.0;
      double u = 0.0;
      //Production Moves
      for(int i=0; i<in.SWEEP; i++)
      {
    	d = Distance2(b[0].coord,b[in.NBEAD-1].coord,in);
		d2 = calcRg(b, in.NBEAD);
		
	  Re+=d;
	  Rg+=d2;
	  u += in.ENERGY;
	  //Choose a particle to move
	  index=gsl_rng_uniform_int(in.gsl_r,in.NBEAD);
	  //Move it
	  Acceptance+=MCMove(b, index,in);
	  
	  if(in.ENERGY>=max_u)
	  	max_u = in.ENERGY;
	  if(in.ENERGY<=min_u)
	  	min_u = in.ENERGY;
	  
	  if(i%in.SAMPLE==0)
		{   
			
		  	d3 = calcRg3D(b, in, r_cm_x,r_cm_y,r_cm_z);
		 	d4 = ShapeAnisotrpy(d3[0],d3[1],d3[2]);
		  	d5 = sideLength(b, in);

		  string f2;
		  f2 = in.SAVE_LOC + "Eq_" + to_string(i)+".txt";
		  out.open(f2,ios::app);	
		
		  /*
		  for(int i =0;i<in.NBEAD;i++)
		    {	
			  out<<b[i].coord[0]<<","<<b[i].coord[1]<<","<<b[i].coord[2]<<endl;
		    }
		  */
		  	out<<"RE,"<<d;
		    out<<"\nRG,"<<d2;
		    out<<"\nInternal Energy,"<<in.ENERGY;
		    out<<"\nRG 3D,"<<d3[0];
		    out<<"\nRG 3D,"<<d3[1];
		    out<<"\nRG 3D,"<<d3[2];
		    out<<"\nShape Anisotropy,"<<d4;
		    out<<"\nSideX,"<<d5[0];
		    out<<"\nSideY,"<<d5[1];
		    out<<"\nSideZ,"<<d5[2];
		    out<<"\nSpringConstant,"<<in.SPRING;
		    for(int i =0;i<in.NBEAD;i++)
		    {	
			 out<<"\n"<<b[i].coord[0]<<","<<b[i].coord[1]<<","<<b[i].coord[2];
		    }
	    
		    out.close();


	    	temp = in.ENERGY;
	        bad_moves(b, i+2, in);
	      	assert(in.ENERGY == temp);
	   
	      }

     }
      start=time ( NULL );
      // Re Vs N
      //string f3 = in.SAVE_LOC + in.NBEADS + "_Final_Values" + ".txt";
      filename = in.SAVE_LOC+ "FinalValues" +".txt";
	    
      out.open(filename,ios::app);
      double eq_u = u/in.SWEEP;
      //out<<"\nEquilibrium Re="<<Re/in.SWEEP<<endl;
      out<<"RE,"<<Re/in.SWEEP<<endl;
      out<<"RG,"<<Rg/in.SWEEP<<endl;
      out<<"Eq U,"<<u/in.SWEEP<<endl;
      if(abs(eq_u-max_u) >= abs(eq_u-min_u))
      	out<<"Best U,"<<max_u;
      else
      	out<<"Best U,"<<min_u;
      out<<endl;
      out<<"Max U,"<<max_u;
      out<<endl;
      out<<"Min U,"<<min_u;
      
      //out<<Rg/in.SWEEP<<endl;
      out.close();      
 }
