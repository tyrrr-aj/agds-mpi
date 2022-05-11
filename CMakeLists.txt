cmake_minimum_required(VERSION 3.5)

project(agds_mpi)

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER mpic++)

include_directories(SYSTEM $ENV{HOME}/installs/mpe2/include)
link_directories($ENV{HOME}/installs/mpe2/lib)

#message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

add_executable(agds_mpi src/main.cpp)

add_library(scan_prod_mpi src/scan_prod_mpi.cpp)
add_library(inference src/inference.cpp)
add_library(utils src/utils.cpp)
add_library(logging_states src/logging_states.cpp)
add_library(mock_agds_data src/mock_agds_data.cpp)

target_link_libraries(agds_mpi scan_prod_mpi inference utils logging_states mock_agds_data mpe pthread)

add_compile_options(-Wall -O3)