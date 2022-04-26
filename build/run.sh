if [ $# -gt 0 ]
then
    make
    mpiexec -n $1 ./agds_mpi
else
    make
    mpiexec -n 16 ./agds_mpi
fi
