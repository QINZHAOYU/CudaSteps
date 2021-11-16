#include <iostream>
#include <typeinfo>
#include "add.cuh"
#include "clock.cuh"

using namespace std;


int main()
{
    if (typeid(double) == typeid(real))
    cout << "using double precision version" << endl;

    cuda_clock();

    return 0;
}


