#ifndef UTILS
#define UTILS

#include <string>
#include <stdio.h>


void cumulated_sum_shifted(int* arr, int len, int* res); // for [1,2,3,4,5] fills res with [0,1,3,6,10]

void print_arr_mpi(double* arr, std::string name, int len, MPI_Comm comm);
void print_arr_mpi(int* arr, std::string name, int len, MPI_Comm comm);

void print_arr(double** arr, std::string name, int len);
void print_arr(double* arr, std::string name, int len);
void print_arr(int* arr, std::string name, int len);

#endif