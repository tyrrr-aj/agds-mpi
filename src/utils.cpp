#include "mpi.h"

#include "utils.hpp"


void cumulated_sum_shifted(int* arr, int len, int* res) { // for [1,2,3,4,5] fills res with [0,1,3,6,10]
    res[0] = 0;
    for (int i = 1; i < len; i++) {
        res[i] = arr[i-1] + res[i-1];
    }
}


void print_arr(double* arr, std::string name, int len) {
    printf("%s: ", name.c_str());
    for (int i = 0; i < len-1; i++) {
        printf("%.5f, ", arr[i]);
    }
    printf("%.5f\n", arr[len-1]);
}

void print_arr(int* arr, std::string name, int len) {
    printf("%s: ", name.c_str());
    for (int i = 0; i < len-1; i++) {
        printf("%d, ", arr[i]);
    }
    if (len > 0) {
        printf("%d\n", arr[len-1]);
    }
    else {
        printf("<empty>\n");
    }
}


void print_arr_mpi(double* arr, std::string name, int len, MPI_Comm comm) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    for (int p = 0; p < size; p++) {
        if (p == rank) {
            printf("(proc %d) ", rank);
            print_arr(arr, name, len);
        }

        MPI_Barrier(comm);
    }
}

void print_arr_mpi(int* arr, std::string name, int len, MPI_Comm comm) {
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    for (int p = 0; p < size; p++) {
        if (p == rank) {
            printf("(proc %d) ", rank);
            print_arr(arr, name, len);
        }

        MPI_Barrier(comm);
    }
}
