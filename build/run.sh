#!/bin/bash


# parse options

n_processes=16 # default process number
no_debug=-1
debug_line=$no_debug # default - no debugging
debug_file="main"

while [ $# -gt 0 ]; do
    case $1 in
        "-n") n_processes=$2
        shift 1
        ;;
        "-d") debug_line=$2
        shift 1
        ;;
        "-df") debug_file=$2
        shift 1
        ;;
        *) echo "Invalid option: $1" >&2
        exit 1
        ;;
    esac

    shift 1
done


# compile code

make


# run code

if [ $? -eq 0 ] # if build was succesfull, run the code
then

    if [ $debug_line -ne $no_debug ]
    then
        mpiexec -n $n_processes xterm -e gdb -ex "break $debug_file.cpp:$debug_line" -ex "run" -ex "print rank" ./agds_mpi
    else
        mpiexec -n $n_processes ./agds_mpi
    fi

else
    echo "Compilation errors - skipping execution"
    exit 1
fi
