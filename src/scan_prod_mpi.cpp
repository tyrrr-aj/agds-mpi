#include "mpi.h"
#include <stdio.h>
#include <cstdlib>
#include <string>

#include "scan_prod_mpi.hpp"
#include "utils.hpp"


void scan_prod_mpi(double* values, int len, double** target, double* vn_range, int* count, int root, MPI_Comm comm) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    double *vals;
    int *sendcounts, *displs;

    if (rank == root) {
        sendcounts = new int[size];
        displs = new int[size];

        int sendcount;
        for (int i = 0; i < size; i++) {
            sendcount = len / size + (i < len % size ? 1 : 0) + (i > 0 ? 1 : 0);
            sendcounts[i] = sendcount;
            displs[i] = i > 0 ? displs[i-1] + sendcounts[i-1] - 1 : 0;
        }

        *vn_range = values[len-1] - values[0];
    }

    MPI_Bcast(vn_range, 1, MPI_DOUBLE, root, comm);

    int send_count;
    MPI_Scatter(sendcounts, 1, MPI_INT, &send_count, 1, MPI_INT, root, comm);
    vals = new double[send_count];
    *count = send_count - (rank > 0 ? 1 : 0);
    *target = new double[*count];

    MPI_Scatterv(values, sendcounts, displs, MPI_DOUBLE, vals, send_count, MPI_DOUBLE, root, comm);

    if (rank == 0) {
        (*target)[0] = 1.0;
        for (int i = 1; i < *count; i++) {
            (*target)[i] = (*vn_range - (vals[i] - vals[i-1])) * (*target)[i-1] / *vn_range;
        }
    }
    else {
        (*target)[0] = (*vn_range - (vals[1] - vals[0])) / *vn_range;
        for (int i = 1; i < *count; i++) {
            (*target)[i] = (*vn_range - (vals[i+1] - vals[i])) * (*target)[i-1] / *vn_range;
        }
    }

    double prev_scan = 1;
    MPI_Exscan(&((*target)[*count - 1]), &prev_scan, 1, MPI_DOUBLE, MPI_PROD, comm);

    for (int i = 0; i < *count; i++) {
        (*target)[i] *= prev_scan;
    }

    delete[] vals;

    if (rank == root) {
        delete[] sendcounts;
        delete[] displs;
    }
}


// testing


int test(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double *values, *prod;

    const int N = 20;
    if (rank == 0) {
        values = new double[N];

        for (int i = 0; i < N; i++) {
            values[i] = i;
        }
    }

    double vn_range;
    int count;

    scan_prod_mpi(values, N, &prod, &vn_range, &count, 0, MPI_COMM_WORLD);

    for (int i = 0; i < size; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == rank) {
            print_arr(prod, "Result", count);
        }
    }

    if (rank == 0) {
        delete[] values;
    }
    delete[] prod;

    return 0;
}


// int main(int argc, char** argv) {
//     return test(argc, argv);
// }
