PROG = MCMIN

SRC = mcmin.cpp

OBJS = ${SRC:.cpp=.o}

CXX = g++
#CXXFLAGS=-g -Wall
GSL_INCLUDE_PATH=/usr/local/include/gsl/
GSL_LIBRARY_PATH=/usr/local/lib/
CXXFLAGS=-O3 -I ${GSL_INCLUDE_PATH} -L ${GSL_LIBRARY_PATH} -lgsl -lgslcblas -g -Wall -lboost_program_options

all: $(PROG)

MC:  	main.o utils.o objutils.o montecarlo.o 
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o:  %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean: 
	rm -f *.o 

distclean:
	rm -f MC *.o *.debug
