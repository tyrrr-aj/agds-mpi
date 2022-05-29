#ifndef UTILS
#define UTILS

#include <string>
#include <vector>
#include <stdio.h>
#include <algorithm>


void cumulated_sum(int* arr, int len, int* res); // for [1,2,3,4,5] fills res with [1,3,6,10,15]
void cumulated_sum_shifted(int* arr, int len, int* res); // for [1,2,3,4,5] fills res with [0,1,3,6,10]
int sum(int* arr, int len);
int max_elem_index(int* arr, int len);
std::vector<int> sort_indices(double* v, int n);

int n_elems_in_equal_split(int total_n_elements, int n_portions, int portion_index);

void print_arr_mpi(double* arr, std::string name, int len, MPI_Comm comm);
void print_arr_mpi(int* arr, std::string name, int len, MPI_Comm comm);

void print_arr(double** arr, std::string name, int len);
void print_arr(double* arr, std::string name, int len);
void print_arr(int* arr, std::string name, int len);

template<typename T>
int upper_bound(T* arr, int len, T value) {
    auto result_it = std::upper_bound(arr, arr + len, value);
    return std::distance(arr, result_it);
}

#endif