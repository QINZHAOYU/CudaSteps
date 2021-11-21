#pragma once
#include <iostream>

using std::cout;
using std::endl;


#ifdef USE_DP
    typedef double real;  
    const real EPSILON = 1.0e-15;
    const char *FLOAT_PREC = "------ double precision";
#else
    typedef float real;   
    const real EPSILON = 1.0e-6f;
    const char *FLOAT_PREC = "------ float precision";
#endif