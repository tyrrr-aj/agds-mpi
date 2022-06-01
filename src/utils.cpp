#include "mpi.h"

#include <vector>
#include <algorithm>
#include <numeric>

#include "utils.hpp"


void cumulated_sum(int* arr, int len, int* res) { // for [1,2,3,4,5] fills res with [1,3,6,10,15]
    res[0] = arr[0];
    for (int i = 1; i < len; i++) {
        res[i] = arr[i] + res[i-1];
    }
}

void cumulated_sum_shifted(int* arr, int len, int* res) { // for [1,2,3,4,5] fills res with [0,1,3,6,10]
    res[0] = 0;
    for (int i = 1; i < len; i++) {
        res[i] = arr[i-1] + res[i-1];
    }
}

int sum(int* arr, int len) {
    int res = 0;
    for (int i = 0; i < len; i++) {
        res += arr[i];
    }

    return res;
}

int max_elem_index(int* arr, int len) {
    int res = 0;
    int max = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] > max) {
            max = arr[i];
            res = i;
        }
    }
    return res;
}


std::vector<int> sort_indices(double* v, int n) {
  std::vector<int> idx(n);
  iota(idx.begin(), idx.end(), 0);

  stable_sort(idx.begin(), idx.end(),
       [v](int i1, int i2) {return v[i1] < v[i2];});

  return idx;
}


int n_elems_in_equal_split(int total_n_elements, int n_portions, int portion_index) {
    // function returns the number of elements assigned to portion i when splitting n (indivisible) elements between p portions in the most equal way possible
    // assuming that if equal split is impossible, portions 0...(n % p) should get n / p + 1 elements, and portions (n % p + 1)...p should get n / p elements
    return total_n_elements / n_portions + ((total_n_elements % n_portions > portion_index) ? 1 : 0);
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
