#ifndef SCAN_PROD
#define SCAN_PROD

#include "mpi.h"

void scan_prod_mpi(double* values, int len, double* &target, double &vn_range, int &count, int root, const MPI_Comm &comm);

#endif