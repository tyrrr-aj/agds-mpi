#include "mpi.h"
#include <stdio.h>
#include <cstdlib>
#include <string>

#include "scan_prod_mpi.hpp"
#include "utils.hpp"


void scan_prod_mpi(double* const values, const int len, double* &target, double &vn_range, int &count, const int root, const MPI_Comm &comm) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    double *vals;
    int *sendcounts, *displs;

    if (rank == root) {
        sendcounts = new int[size];
        displs = new int[size];

        sendcounts[0] = n_elems_in_equal_split(len, size, 0);
        displs[0] = 0;

        for (int i = 1; i < size; i++) {
            sendcounts[i] = n_elems_in_equal_split(len, size, i) + 1; // +1 because each process needs predecessor to its first value
            displs[i] = displs[i-1] + sendcounts[i-1] - 1; // -1 for the same reason as above
        }

        vn_range = values[len-1] - values[0];
    }

    MPI_Bcast(&vn_range, 1, MPI_DOUBLE, root, comm);

    int send_count;
    MPI_Scatter(sendcounts, 1, MPI_INT, &send_count, 1, MPI_INT, root, comm);

    vals = new double[send_count];
    MPI_Scatterv(values, sendcounts, displs, MPI_DOUBLE, vals, send_count, MPI_DOUBLE, root, comm);

    count = send_count - (rank > 0 ? 1 : 0); // all but the first process received one more value then tey're supposed to compute
    target = new double[count];
    double* shifted_vals;

    if (rank == 0) {
        target[0] = 1.0;
        shifted_vals = vals; // process received exactly the same number of values as it's supposed to compute
    }
    else {
        target[0] = (vn_range - (vals[1] - vals[0])) / vn_range; // here the additional value (vals[0]) is necessary
        shifted_vals = vals + 1; // process received one additional value at the beginning of vals which should be ignored from now on
    }

    for (int i = 1; i < count; i++) {
        target[i] = (vn_range - (shifted_vals[i] - shifted_vals[i-1])) * target[i-1] / vn_range;
    }

    double prev_scan = 1;
    MPI_Exscan(target + count - 1, &prev_scan, 1, MPI_DOUBLE, MPI_PROD, comm);

    for (int i = 0; i < count; i++) {
        target[i] *= prev_scan;
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

    scan_prod_mpi(values, N, prod, vn_range, count, 0, MPI_COMM_WORLD);

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
