if [ $# -gt 0 ]
then
    make
    mpiexec -n $1 xterm -e gdb ./agds_mpi
else
    make
    mpiexec -n 16 xterm -e gdb ./agds_mpi
fi
