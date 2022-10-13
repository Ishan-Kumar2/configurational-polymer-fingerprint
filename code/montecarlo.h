#ifndef _MONTECARLO_H
#define _MONTECARLO_H
void bad_moves(vector<Bead>& b, int i, Input& in, bool mul);
int MCMove(vector<Bead>& , int, Input& );
void MonteCarlo(vector<Bead>&, Input& );
double calcRg(vector<Bead>& , int);
#endif
