#include <mpi.h>
#include <mpe.h>

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPE_Init_log();

    int event_start, event_end;
    MPE_Log_get_state_eventIDs(&event_start, &event_end);
    MPE_Describe_state(event_start, event_end, "empire state of mind", "red");

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPE_Log_event(event_start, 0, "alfa");

    std::chrono::milliseconds timespan(rank * 125);
    std::this_thread::sleep_for(timespan);
    printf("Działa w procesie %d\n", rank);

    int* a = new int[5];
    delete[] a;

    MPE_Log_event(event_end, 0, "omega");

    MPE_Finish_log("test");
    MPI_Finalize();
}