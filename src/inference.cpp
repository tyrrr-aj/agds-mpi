#include <mpi.h>
#include <mpe.h>

#include "inference.hpp"
#include "utils.hpp"
#include "events.hpp"
#include "logging_states.hpp"


int N_REC;

int N_VN_PROC;
int N_ON_PROC;

int N_VN_IN_VNG;

int* VNG_SIZES; // number of processes, not VNs
int* VNG_SIZES_CUMULATED; // number of processes, not VNs

int own_vng_id;

MPI_Comm VNG_COMM;

int* CONN;

// locally initialized
double* VN_A;
double* VN_R;
int* VN_N;
double* VN_A_FROM_ONs;
double* VN_A_WEIGHTED;
double* ON_A;

double* OWN_STARTING_POINTS_R;
double* OWN_STARTING_POINTS_A;
double* STARTING_POINTS_R;
double* STARTING_POINTS_A;

int N_OWN_STARTING_POINTS;
int* N_STARTING_POINTS_IN_PROCESSES;
int* DISPLS;

int VNG_N_PROC;

MPI_Win win_on2vn;
MPI_Win win_vn2on;


double weight_vns(double r_left, double r_right) {
    return r_left < r_right ? r_left / r_right : r_right / r_left;
}


int vn_conn_proc_id(int n_proc, int n_vn_conns, int conn_global_id) {
    int max_n_conns_per_proc = n_vn_conns / n_proc + 1;
    int min_n_conns_per_proc = n_vn_conns / n_proc;
    
    int threshold = n_vn_conns % n_proc;

    if (conn_global_id / max_n_conns_per_proc < threshold) {
        return conn_global_id / max_n_conns_per_proc;
    }
    else {
        return threshold + (conn_global_id - threshold * max_n_conns_per_proc) / min_n_conns_per_proc;
    }
}


int vn_proc_id(int vn_id, int vng_id) {
    return VNG_SIZES_CUMULATED[vng_id] + vn_id % VNG_SIZES[vng_id];
}

int vn_local_ix(int vn_id, int vng_id) {
    return vn_id / VNG_SIZES[vng_id] + 1;
}


void allocate_mem_for_starting_points(int* displs) {
    int n_starting_points = displs[VNG_N_PROC - 1] + N_STARTING_POINTS_IN_PROCESSES[VNG_N_PROC - 1];

    STARTING_POINTS_R = new double[n_starting_points];
    STARTING_POINTS_A = new double[n_starting_points];
}

void free_mem_for_starting_points() {
    // MPE_Log_event(ON2VN_FREE_MEMORY_START, 0, "");

    delete[] STARTING_POINTS_R;
    delete[] STARTING_POINTS_A;

    // MPE_Log_event(ON2VN_FREE_MEMORY_END, 0, "");
}

void calculate_weighted_vn_activations() {
    for (int vn_ix = 0; vn_ix < N_VN_PROC; vn_ix++) {
        VN_A_WEIGHTED[vn_ix] = VN_A[vn_ix] / VN_N[vn_ix];
    }
}

void get_activations_from_vns() {
    MPI_Win_fence(MPI_MODE_NOPRECEDE, win_vn2on);

    double remote_vn_act;
    for (int on_ix = 0; on_ix < N_ON_PROC; on_ix++) {
        for (int vng_ix = 0; vng_ix < N_REC; vng_ix++) {
            if (CONN[on_ix * N_REC + vng_ix] != -1) {
                MPI_Get(
                    &remote_vn_act, 
                    1, 
                    MPI_DOUBLE, 
                    vn_proc_id(CONN[on_ix * N_REC + vng_ix], vng_ix), 
                    vn_local_ix(CONN[on_ix * N_REC + vng_ix], vng_ix), 
                    1, 
                    MPI_DOUBLE,
                    win_vn2on
                );

                ON_A[on_ix] += remote_vn_act;
            }
        }
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win_vn2on);
}



void distribute_vn_starting_points() {
    // MPE_Log_event(ON2VN_DISTRIBUTE_VN_STARTING_POINTS_START, 0, "");

    MPI_Allgather(&N_OWN_STARTING_POINTS, 1, MPI_INT, N_STARTING_POINTS_IN_PROCESSES, 1, MPI_INT, VNG_COMM);
    
    cumulated_sum_shifted(N_STARTING_POINTS_IN_PROCESSES, VNG_N_PROC, DISPLS);
    allocate_mem_for_starting_points(DISPLS);

    MPI_Allgatherv(OWN_STARTING_POINTS_R, N_OWN_STARTING_POINTS, MPI_DOUBLE, STARTING_POINTS_R, N_STARTING_POINTS_IN_PROCESSES, DISPLS, MPI_DOUBLE, VNG_COMM);
    MPI_Allgatherv(OWN_STARTING_POINTS_A, N_OWN_STARTING_POINTS, MPI_DOUBLE, STARTING_POINTS_A, N_STARTING_POINTS_IN_PROCESSES, DISPLS, MPI_DOUBLE, VNG_COMM);

    // MPE_Log_event(ON2VN_DISTRIBUTE_VN_STARTING_POINTS_END, 0, "");
}

void calculate_vn_activations() {
    // MPE_Log_event(ON2VN_CALCULATE_VN_ACTIVATIONS_START, 0, "");

    for (int vn_ix = 0; vn_ix < N_VN_PROC; vn_ix++) {
        for (int sp_ix = 0; sp_ix < N_OWN_STARTING_POINTS; sp_ix++) {
            VN_A[vn_ix] += weight_vns(VN_R[vn_ix], STARTING_POINTS_R[sp_ix]) * STARTING_POINTS_A[sp_ix];
        }
    }

    // MPE_Log_event(ON2VN_CALCULATE_VN_ACTIVATIONS_END, 0, "");
}


void update_vns_activated_by_ons() {
    // MPE_Log_event(ON2VN_UPDATE_VNS_ACTIVATED_BY_ONS_START, 0, "");

    for (int i = 0; i < N_VN_PROC; i++) {
        VN_A[i] += VN_A_FROM_ONs[i];
    }

    // MPE_Log_event(ON2VN_UPDATE_VNS_ACTIVATED_BY_ONS_END, 0, "");
}

void find_own_starting_points() {
    // MPE_Log_event(ON2VN_FIND_OWN_STARTING_POINTS_START, 0, "");
    
    N_OWN_STARTING_POINTS = 0;
    for (int vn_ix = 0; vn_ix < N_VN_PROC; vn_ix++) {
        if (VN_A[vn_ix] > 0) {
            OWN_STARTING_POINTS_R[N_OWN_STARTING_POINTS] = VN_R[vn_ix];
            OWN_STARTING_POINTS_A[N_OWN_STARTING_POINTS] = VN_A[vn_ix];

            N_OWN_STARTING_POINTS++;
        }
    }

    // MPE_Log_event(ON2VN_FIND_OWN_STARTING_POINTS_END, 0, "");
}


void send_activations_from_on_to_vn() {
    // MPE_Log_event(ON2VN_SEND_ACTIVATIONS_START, 0, "");

    MPI_Win_fence(MPI_MODE_NOPRECEDE, win_on2vn);

    // printf("I've got %d ONs to process\n", N_ON_PROC);

    // for (int on_ix = 0; on_ix < N_ON_PROC; on_ix++) {
    for (int vng_ix = 0; vng_ix < N_REC; vng_ix++) {
        // MPE_Log_event(SEND_ACTIVATIONS_SEND_VNG_START, 0, "");

        // for (int vng_ix = 0; vng_ix < N_REC; vng_ix++) {
        for (int on_ix = 0; on_ix < N_ON_PROC; on_ix++) {
            // MPE_Log_event(SEND_ACTIVATIONS_SEND_SINGLE_ON_START, 0, "");

            // printf("CONN[on_ix * N_REC + vng_ix]: %d, vn_local_ix(): %d\n", CONN[on_ix * N_REC + vng_ix], vn_local_ix(CONN[on_ix * N_REC + vng_ix], vng_ix));
            if (CONN[on_ix * N_REC + vng_ix] != -1) {
                MPI_Accumulate(
                    &ON_A[on_ix], 
                    1, 
                    MPI_DOUBLE, 
                    vn_proc_id(CONN[on_ix * N_REC + vng_ix], vng_ix), 
                    vn_local_ix(CONN[on_ix * N_REC + vng_ix], vng_ix),
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    win_on2vn
                );
            }

            // MPE_Log_event(SEND_ACTIVATIONS_SEND_SINGLE_ON_END, 0, "");
        }

        // MPE_Log_event(SEND_ACTIVATIONS_SEND_VNG_END, 0, "");
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win_on2vn);

    // MPE_Log_event(ON2VN_SEND_ACTIVATIONS_END, 0, "");
}


void vn2on_step() {
    MPE_Log_event(INFERENCE_VN2ON_START, 0, "");

    calculate_weighted_vn_activations();
    get_activations_from_vns();

    MPE_Log_event(INFERENCE_VN2ON_END, 0, "");
}

void on2vn_step() {
    MPE_Log_event(INFERENCE_ON2VN_START, 0, "");

    send_activations_from_on_to_vn();
    update_vns_activated_by_ons();
    find_own_starting_points();
    distribute_vn_starting_points();
    calculate_vn_activations();
    free_mem_for_starting_points();

    MPE_Log_event(INFERENCE_ON2VN_END, 0, "");
}

void setup_for_inference(int n_vn_p, int n_on_p, int n_groups, int* vng_sizes, int n_vn_vng, int vng_ix, MPI_Comm vng_comm, int* conn, double* Vn_r, int* Vn_n) {
    N_REC = n_groups;
    N_VN_PROC = n_vn_p;
    N_ON_PROC = n_on_p;
    VNG_SIZES = vng_sizes;

    VNG_SIZES_CUMULATED = new int[n_groups];
    cumulated_sum_shifted(vng_sizes, n_groups, VNG_SIZES_CUMULATED);\

    N_VN_IN_VNG = n_vn_vng;
    own_vng_id = vng_ix;
    VNG_COMM = vng_comm;
    CONN = conn;
    VN_R = Vn_r;
    VN_N = Vn_n;

    MPI_Comm_size(vng_comm, &VNG_N_PROC);

    VN_A = new double[N_VN_PROC];
    ON_A = new double[N_ON_PROC];

    OWN_STARTING_POINTS_R = new double[N_VN_PROC];
    OWN_STARTING_POINTS_A = new double[N_VN_PROC];

    N_STARTING_POINTS_IN_PROCESSES = new int[VNG_N_PROC];
    DISPLS = new int[VNG_N_PROC];

    VN_A_FROM_ONs = new double[n_vn_p];
    VN_A_WEIGHTED = new double[n_vn_p];
    
    MPI_Win_allocate(n_vn_p * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &VN_A_FROM_ONs, &win_on2vn);
    MPI_Win_allocate(n_vn_p * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &VN_A_WEIGHTED, &win_vn2on);
}


void teardown_inference() {
    MPI_Win_free(&win_on2vn);
    MPI_Win_free(&win_vn2on);

    delete[] VN_A;
    delete[] ON_A;

    delete[] OWN_STARTING_POINTS_R;
    delete[] OWN_STARTING_POINTS_A;

    delete[] N_STARTING_POINTS_IN_PROCESSES;
    delete[] DISPLS;
}


void init_inference(int* activated_vns, int n_activated_vns, int* activated_ons, int n_activated_ons) {
    for (int vn_ix = 0; vn_ix < N_VN_PROC; vn_ix++) {
        VN_A[vn_ix] = 0;
    }
    
    for (int act_vn_ix = 0; act_vn_ix < n_activated_vns; act_vn_ix++) {
        VN_A[activated_vns[act_vn_ix]] = 1;
    }

    for (int on_ix = 0; on_ix < N_ON_PROC; on_ix++) {
        ON_A[on_ix] = 0;
    }

    for (int act_on_ix = 0; act_on_ix < n_activated_ons; act_on_ix++) {
        ON_A[activated_ons[act_on_ix] % N_ON_PROC] = 1;
    }
}

void init_step() {
    MPE_Log_event(INFERENCE_INIT_START, 0, "");

    for (int vn_ix = 0; vn_ix < N_VN_PROC; vn_ix++) {
        VN_A_FROM_ONs[vn_ix] = 0;
    }

    MPE_Log_event(INFERENCE_INIT_END, 0, "");
}

void inference(int* activated_vns, int n_activated_vns, int* activated_ons, int n_activated_ons, bool vng_in_query, int steps) {
    init_inference(activated_vns, n_activated_vns, activated_ons, n_activated_ons);

    for (int s = 0; s < steps; s++) {
        init_step();
        on2vn_step();
        vn2on_step();
    }

    // print_arr_mpi(VN_A, "VN_A", N_VN_PROC, MPI_COMM_WORLD);
    // print_arr_mpi(ON_A, "ON_A", N_ON_PROC, MPI_COMM_WORLD);
}
