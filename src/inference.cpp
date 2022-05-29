#include <mpi.h>
#include <mpe.h>

#include <algorithm>

#include "inference.hpp"
#include "utils.hpp"
#include "events.hpp"
#include "logging_states.hpp"


// externally initialized

int N_PROC;

int N_GROUPS;

int N_VN_PROC;
int N_ON_PROC;

int N_CONN_GLOBAL; // total number of ON->VN connections (in all processes)

int N_CONN_PROC; // number of INCOMING ON->VN connections controlled by the process; = sum(VNS_N_CONNS)
int N_VN_CONN_PROC; // number of distinct VNs whose connections are controlled by the process

int* VNS_N_CONNS; // number of ON->VN connections for each VN controlled by the process at the ON->VN step

/* each VN controlled by process may be distributed (for processing ON->VN connections) over multiple 
processes; this array stores the number of processes over which each VN controlled by the process
is distributed */
int* VNS_DISTRIBUTION;
int* VNS_PROC; // global indices of VNs controlled by the process in VN->VN and VN->ON step
int* VNS_PROC_ON_TO_VN; // global indices of VNs controlled by the process in ON->VN step

int N_VN_IN_VNG;

int* VNG_SIZES_PROC; // number of processes responsible for each VNG
int* VNG_SIZES_PROC_CUMULATED_SHIFTED;

int* VNG_SIZES_VNS; // number of VNs belonging to each VNG
int* VNG_SIZES_VNS_CUMULATED;
int* VNG_SIZES_VNS_CUMULATED_SHIFTED;

int own_vng_id;

MPI_Comm VNG_COMM;

int* CONN; // ids of VNs to which each ON is connected, stored by ONs and VNGs ([CONN_ON0_VNG0, CONN_ON0_VNG1, ..., CONN_ON1_VNG0, ...])
int* CONN_global_ix; // global IDs of ON->VN connections, stored by ONs and VNGs ([CONN_ON0_VNG0, CONN_ON0_VNG1, ..., CONN_ON1_VNG0, ...])


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

// MPI_Win win_on2vn;
MPI_Win win_vn2on;

MPI_Request* on_to_vn_send_requests;
bool are_on_to_vn_send_requests_filled;

MPI_Request* vn_accumulation_send_requests;
bool are_vn_accumulation_send_requests_filled;

double weight_vns(double r_left, double r_right) {
    return r_left < r_right ? r_left / r_right : r_right / r_left;
}


#pragma region Finding IDs and local indices of processes responsible for specific VNs (in VN->VN and VN->ON steps)

int vn_proc_id(int vn_id, int vng_id) {
    return VNG_SIZES_PROC_CUMULATED_SHIFTED[vng_id] + vn_id % VNG_SIZES_PROC[vng_id];
}

int vn_local_ix(int vn_id, int vng_id) {
    return vn_id / VNG_SIZES_PROC[vng_id] + 1;
}

#pragma endregion


#pragma region Finding IDs and local indices of processes responsible for connections ON->VN

int vn_conn_proc_id(int conn_global_id) {
    int max_n_conns_per_proc = N_CONN_GLOBAL / N_PROC + 1;
    int min_n_conns_per_proc = N_CONN_GLOBAL / N_PROC;
    
    int threshold = N_CONN_GLOBAL % N_PROC;

    if (conn_global_id / max_n_conns_per_proc < threshold) {
        return conn_global_id / max_n_conns_per_proc;
    }
    else {
        return threshold + (conn_global_id - threshold * max_n_conns_per_proc) / min_n_conns_per_proc;
    }
}

int vn_conn_local_idx(int dest_proc_id, int conn_global_id) {
    int max_n_conns_per_proc = N_CONN_GLOBAL / N_PROC + 1;
    int min_n_conns_per_proc = N_CONN_GLOBAL / N_PROC;
    
    int threshold = N_CONN_GLOBAL % N_PROC;

    if (dest_proc_id < threshold) {
        return conn_global_id - (dest_proc_id * max_n_conns_per_proc);
    }
    else {
        return conn_global_id - (threshold * max_n_conns_per_proc) - (dest_proc_id - threshold) * min_n_conns_per_proc;
    }
}

int vn_by_conn_local_id(int conn_local_idx) {
    return upper_bound(VNS_N_CONNS, N_VN_CONN_PROC, conn_local_idx);
}

int vn_proc_id(int vn_global_id) {
    int vng_id = upper_bound(VNG_SIZES_VNS_CUMULATED, N_GROUPS, vn_global_id);
    int vn_local_id = vn_global_id - VNG_SIZES_VNS_CUMULATED_SHIFTED[vng_id];
    
    return vn_proc_id(vn_local_id, vng_id);
}

#pragma endregion


#pragma region Starting points in VN->VN step

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

void distribute_vn_starting_points() {
    // MPE_Log_event(ON2VN_DISTRIBUTE_VN_STARTING_POINTS_START, 0, "");

    MPI_Allgather(&N_OWN_STARTING_POINTS, 1, MPI_INT, N_STARTING_POINTS_IN_PROCESSES, 1, MPI_INT, VNG_COMM);
    
    cumulated_sum_shifted(N_STARTING_POINTS_IN_PROCESSES, VNG_N_PROC, DISPLS);
    allocate_mem_for_starting_points(DISPLS);

    MPI_Allgatherv(OWN_STARTING_POINTS_R, N_OWN_STARTING_POINTS, MPI_DOUBLE, STARTING_POINTS_R, N_STARTING_POINTS_IN_PROCESSES, DISPLS, MPI_DOUBLE, VNG_COMM);
    MPI_Allgatherv(OWN_STARTING_POINTS_A, N_OWN_STARTING_POINTS, MPI_DOUBLE, STARTING_POINTS_A, N_STARTING_POINTS_IN_PROCESSES, DISPLS, MPI_DOUBLE, VNG_COMM);

    // MPE_Log_event(ON2VN_DISTRIBUTE_VN_STARTING_POINTS_END, 0, "");
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

#pragma endregion


#pragma region VN activations in VN->VN step

void calculate_weighted_vn_activations() {
    for (int vn_ix = 0; vn_ix < N_VN_PROC; vn_ix++) {
        VN_A_WEIGHTED[vn_ix] = VN_A[vn_ix] / VN_N[vn_ix];
    }
}

void get_activations_from_vns() {
    if (are_on_to_vn_send_requests_filled) {
        MPI_Waitall(N_ON_PROC * N_GROUPS, on_to_vn_send_requests, MPI_STATUSES_IGNORE);
        are_on_to_vn_send_requests_filled = false;
    }

    MPI_Win_fence(MPI_MODE_NOPRECEDE, win_vn2on);

    double remote_vn_act;
    for (int on_ix = 0; on_ix < N_ON_PROC; on_ix++) {
        for (int vng_ix = 0; vng_ix < N_GROUPS; vng_ix++) {
            if (CONN[on_ix * N_GROUPS + vng_ix] != -1) {
                MPI_Get(
                    &remote_vn_act, 
                    1, 
                    MPI_DOUBLE, 
                    vn_proc_id(CONN[on_ix * N_GROUPS + vng_ix], vng_ix), 
                    vn_local_ix(CONN[on_ix * N_GROUPS + vng_ix], vng_ix), 
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

void calculate_vn_activations() {
    // MPE_Log_event(ON2VN_CALCULATE_VN_ACTIVATIONS_START, 0, "");

    for (int vn_ix = 0; vn_ix < N_VN_PROC; vn_ix++) {
        for (int sp_ix = 0; sp_ix < N_OWN_STARTING_POINTS; sp_ix++) {
            VN_A[vn_ix] += weight_vns(VN_R[vn_ix], STARTING_POINTS_R[sp_ix]) * STARTING_POINTS_A[sp_ix];
        }
    }

    // MPE_Log_event(ON2VN_CALCULATE_VN_ACTIVATIONS_END, 0, "");
}

#pragma endregion


#pragma region Accumulating VN activations at the end of ON->VN step

void setup_receiving_vn_accumulation(MPI_Request* vn_accumulation_receive_requests, double* vn_accumulation_receive_buffer) {
    int idx = 0;

    for (int i = 0; i < N_VN_PROC; i++) {
        for (int j = 0; j < VNS_DISTRIBUTION[i]; j++) {
            MPI_Irecv(
                &vn_accumulation_receive_buffer[idx],
                1,
                MPI_DOUBLE,
                MPI_ANY_SOURCE,
                VNS_PROC[i],
                MPI_COMM_WORLD,
                &vn_accumulation_receive_requests[i]);

            idx++;
        }
    }
}

void send_vn_accumulation(double* vn_partial_activations) {
    // MPE_Log_event(ON2VN_UPDATE_VNS_ACTIVATED_BY_ONS_START, 0, "");

    for (int i = 0; i < N_VN_CONN_PROC; i++) {
        MPI_Isend(
            &vn_partial_activations[i],
            1,
            MPI_DOUBLE,
            vn_proc_id(VNS_PROC_ON_TO_VN[i]),
            VNS_PROC_ON_TO_VN[i],
            MPI_COMM_WORLD,
            &vn_accumulation_send_requests[i]);
    }

    are_vn_accumulation_send_requests_filled = true;

    // MPE_Log_event(ON2VN_UPDATE_VNS_ACTIVATED_BY_ONS_END, 0, "");
}

void receive_and_accumulate_vn_activations(MPI_Request* vn_accumulation_receive_requests, double* vn_accumulation_receive_buffer) {
    int n_requests = sum(VNS_DISTRIBUTION, N_VN_PROC);
    int* cumulated_vn_distribution = new int[N_VN_PROC];
    int index;

    cumulated_sum(VNS_DISTRIBUTION, N_VN_PROC, cumulated_vn_distribution);

    for (int i = 0; i < n_requests; i++) {
        MPI_Waitany(n_requests, vn_accumulation_receive_requests, &index, MPI_STATUS_IGNORE);
        int vn_local_idx = upper_bound(cumulated_vn_distribution, N_VN_PROC, index);
        VN_A[vn_local_idx] += vn_accumulation_receive_buffer[index];
    }

    delete[] cumulated_vn_distribution;
}


void accumulate_vn_activations(double* vn_partial_activations) {
    MPI_Request* vn_accumulation_receive_requests = new MPI_Request[sum(VNS_DISTRIBUTION, N_VN_PROC)];
    double* vn_accumulation_receive_buffer = new double[sum(VNS_DISTRIBUTION, N_VN_PROC)];

    setup_receiving_vn_accumulation(vn_accumulation_receive_requests, vn_accumulation_receive_buffer);
    send_vn_accumulation(vn_partial_activations);
    receive_and_accumulate_vn_activations(vn_accumulation_receive_requests, vn_accumulation_receive_buffer);

    delete[] vn_accumulation_receive_requests;
    delete[] vn_accumulation_receive_buffer;
}

#pragma endregion


#pragma region Processing of connections in ON->VN step

void setup_receiving_on_to_vn(MPI_Request* on_to_vn_receive_requests, double* on_to_vn_activations_buffer) {
    for (int i = 0; i < N_CONN_PROC; i++) {
        MPI_Irecv(
            &on_to_vn_activations_buffer[i],
            1, 
            MPI_DOUBLE,
            MPI_ANY_SOURCE,
            i,
            MPI_COMM_WORLD, 
            &on_to_vn_receive_requests[i]
        );
    }
}

void send_on_to_vn() {
    // MPE_Log_event(ON2VN_SEND_ACTIVATIONS_START, 0, "");
    int dest_proc_id, dest_local_idx;

    for (int vng_ix = 0; vng_ix < N_GROUPS; vng_ix++) {
        // MPE_Log_event(SEND_ACTIVATIONS_SEND_VNG_START, 0, "");

        for (int on_ix = 0; on_ix < N_ON_PROC; on_ix++) {
            // MPE_Log_event(SEND_ACTIVATIONS_SEND_SINGLE_ON_START, 0, "");

            dest_proc_id = vn_conn_proc_id(CONN_global_ix[on_ix * N_GROUPS + vng_ix]);
            dest_local_idx = vn_conn_local_idx(dest_proc_id, CONN_global_ix[on_ix * N_GROUPS + vng_ix]);

            MPI_Isend(
                &ON_A[on_ix], 
                1, 
                MPI_DOUBLE, 
                dest_proc_id, 
                dest_local_idx, 
                MPI_COMM_WORLD, 
                &on_to_vn_send_requests[on_ix * N_GROUPS + vng_ix]);

            // MPE_Log_event(SEND_ACTIVATIONS_SEND_SINGLE_ON_END, 0, "");
        }

        // MPE_Log_event(SEND_ACTIVATIONS_SEND_VNG_END, 0, "");
    }

    are_on_to_vn_send_requests_filled = true;
    // MPE_Log_event(ON2VN_SEND_ACTIVATIONS_END, 0, "");
}

void receive_on_to_vn(MPI_Request* on_to_vn_receive_requests, double* vn_partial_activations, double* on_to_vn_activations_buffer) {
    int index;
    MPI_Status status;

    if (are_vn_accumulation_send_requests_filled) {
        MPI_Waitall(N_VN_CONN_PROC, vn_accumulation_send_requests, MPI_STATUSES_IGNORE);
        are_vn_accumulation_send_requests_filled = false;
    }

    for (int i = 0; i < N_VN_CONN_PROC; i++) {
        vn_partial_activations[i] = 0;
    }

    for (int i = 0; i < N_CONN_PROC; i++) {
        MPI_Waitany(N_CONN_PROC, on_to_vn_receive_requests, &index, &status);
        vn_partial_activations[vn_by_conn_local_id(index)] += on_to_vn_activations_buffer[index];
    }
}

void process_on_to_vn_connections(double* vn_partial_activations) {
    MPI_Request* on_to_vn_receive_requests = new MPI_Request[N_CONN_PROC];
    double* on_to_vn_activations_buffer = new double[N_CONN_PROC];

    setup_receiving_on_to_vn(on_to_vn_receive_requests, on_to_vn_activations_buffer);
    send_on_to_vn();
    receive_on_to_vn(on_to_vn_receive_requests, vn_partial_activations, on_to_vn_activations_buffer);

    delete[] on_to_vn_receive_requests;
    delete[] on_to_vn_activations_buffer;
}

#pragma endregion


#pragma region Main steps

void vn2on_step() {
    // MPE_Log_event(INFERENCE_VN2ON_START, 0, "");

    calculate_weighted_vn_activations();
    get_activations_from_vns();

    // MPE_Log_event(INFERENCE_VN2ON_END, 0, "");
}

void vn2vn_step() {
    find_own_starting_points();
    distribute_vn_starting_points();
    calculate_vn_activations();
    free_mem_for_starting_points();
}

void on2vn_step() {
    // MPE_Log_event(INFERENCE_ON2VN_START, 0, "");

    double* vn_partial_activations = new double[N_VN_CONN_PROC];

    process_on_to_vn_connections(vn_partial_activations);
    accumulate_vn_activations(vn_partial_activations);

    delete[] vn_partial_activations;

    // MPE_Log_event(INFERENCE_ON2VN_END, 0, "");
}

#pragma endregion


#pragma region Main setup, teardown and public API

void setup_for_inference(int n_vn_p, int n_on_p, int n_groups, int* vng_sizes_proc, int* vng_sizes_vns, int n_vn_vng, int vng_ix, MPI_Comm vng_comm, int* conn, int* conn_global_ix, double* Vn_r, int* Vn_n,
                        int n_conn_global, int n_conn_proc, int n_vn_conn_proc, int* vns_n_conns, int* vns_distribution, int* vns_proc, int* vns_proc_on_to_vn) { // modify CONN to new way!

    N_GROUPS = n_groups;
    N_VN_PROC = n_vn_p;
    N_ON_PROC = n_on_p;
    VNG_SIZES_PROC = vng_sizes_proc;
    VNG_SIZES_VNS = vng_sizes_vns;

    N_CONN_GLOBAL = n_conn_global;
    N_CONN_PROC = n_conn_proc;
    N_VN_CONN_PROC = n_vn_conn_proc;

    VNS_N_CONNS = vns_n_conns;
    VNS_DISTRIBUTION = vns_distribution;
    VNS_PROC = vns_proc;
    VNS_PROC_ON_TO_VN = vns_proc_on_to_vn;

    MPI_Comm_size(MPI_COMM_WORLD, &N_PROC);

    VNG_SIZES_PROC_CUMULATED_SHIFTED = new int[n_groups];
    cumulated_sum_shifted(vng_sizes_proc, n_groups, VNG_SIZES_PROC_CUMULATED_SHIFTED);

    VNG_SIZES_VNS_CUMULATED = new int[n_groups];
    cumulated_sum(vng_sizes_vns, n_groups, VNG_SIZES_VNS_CUMULATED);

    VNG_SIZES_VNS_CUMULATED_SHIFTED = new int[n_groups];
    cumulated_sum_shifted(vng_sizes_vns, n_groups, VNG_SIZES_VNS_CUMULATED);

    N_VN_IN_VNG = n_vn_vng;
    own_vng_id = vng_ix;
    VNG_COMM = vng_comm;
    CONN = conn;
    CONN_global_ix = conn_global_ix;
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

    // MPI_Win_allocate(n_vn_p * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &VN_A_FROM_ONs, &win_on2vn);
    MPI_Win_allocate(n_vn_p * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &VN_A_WEIGHTED, &win_vn2on);

    on_to_vn_send_requests = new MPI_Request[N_ON_PROC * N_GROUPS];
    vn_accumulation_send_requests = new MPI_Request[N_VN_CONN_PROC];
}

void teardown_inference() {
    // MPI_Win_free(&win_on2vn);
    MPI_Win_free(&win_vn2on);

    delete[] VN_A;
    delete[] ON_A;

    delete[] OWN_STARTING_POINTS_R;
    delete[] OWN_STARTING_POINTS_A;

    delete[] N_STARTING_POINTS_IN_PROCESSES;
    delete[] DISPLS;

    delete[] on_to_vn_send_requests;
    delete[] vn_accumulation_send_requests;

    delete[] VNG_SIZES_PROC_CUMULATED_SHIFTED;
    delete[] VNG_SIZES_VNS_CUMULATED;
    delete[] VNG_SIZES_VNS_CUMULATED_SHIFTED;
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

    are_on_to_vn_send_requests_filled = false;
    are_vn_accumulation_send_requests_filled = false;
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
        vn2vn_step();
        vn2on_step();
    }

    // print_arr_mpi(VN_A, "VN_A", N_VN_PROC, MPI_COMM_WORLD);
    // print_arr_mpi(ON_A, "ON_A", N_ON_PROC, MPI_COMM_WORLD);
}

#pragma endregion
