#include <mpi.h>
#include <mpe.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <vector>

#include "scan_prod_mpi.hpp"
#include "inference.hpp"
#include "utils.hpp"
#include "logging_states.hpp"
#include "mock_agds_data.hpp"


const int N_ON = 100000;
const int N_GROUPS = 2;

const int N_QUERIES = 50;
const int ACTIVATED_VNS_PER_GROUP = 2;
const int ACTIVATED_VNGS = N_GROUPS - 1;
const int ACTIVATED_ONS = 5;
const int K = 1;

const int SEED = 42;

const int MASTERS_TAG = 0;
const int AGDS_MASTER_RANK = 0;

const double EPSILON = 0.00001;

#pragma region Debug

void debug_printf_from_vng_leaders(const int rank, const int vng_rank, const int vng_id, const int n_vn_vng) {
    if (vng_rank == 0) {
        printf("I'm %d, 0 in group %d, and my VNG consists of %d VNs\n", rank, vng_id, n_vn_vng);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void debug_printf_from_all_workers(const int rank, const int vng_id, const int n_vn_p) {
    printf("I'm %d, working for %d, and I've received %d values\n", rank, vng_id, n_vn_p);
}

#pragma endregion


#pragma region Generic functions

void mpi_size_and_rank(const MPI_Comm &comm, int &size, int &rank) {
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
}

#pragma endregion


#pragma region VNG group

void split_processes_into_vngs(int* const worker_spread, const int rank, int &vng_id, int &vng_size, int &vng_rank, MPI_Comm &vng_comm) {
    MPI_Scatter(worker_spread, 1, MPI_INT, &vng_id, 1, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
    
    MPI_Comm_split(MPI_COMM_WORLD, vng_id, rank, &vng_comm);
    MPI_Comm_size(vng_comm, &vng_size);
    MPI_Comm_rank(vng_comm, &vng_rank);
}

void broadcast_number_of_vns_in_vngs(int& n_vn_vng, const MPI_Comm &vng_comm) {
    // Each VNG leader informs the rest of processes in VNG about the number of VNs in this VNG
    MPI_Bcast(&n_vn_vng, 1, MPI_INT, 0, vng_comm);
}

void compute_vn_prod(double* const tree, const int n_vn_vng, double* &Vn_prod_from_scan_p, double &vn_range, int &n_vn_p, const MPI_Comm &vng_comm) {
    // compute VN_prod_from_scan_p vector via scan - it's sorted, which is not the desired order
    scan_prod_mpi(tree, n_vn_vng, &Vn_prod_from_scan_p, &vn_range, &n_vn_p, 0, vng_comm);
}

void broadcast_number_of_vns_in_vngs(int* const Vng_n_vns) {
    // Vector with numbers of VNs in each VNG is shared with all processes (by master)
    MPI_Bcast(Vng_n_vns, N_GROUPS, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
}

void gather_number_of_vns_per_proc(const int n_vn_p, int* const n_vn_of_ps_in_vng, int* &displ_vng_p, int vng_rank, int vng_size, const MPI_Comm &vng_comm) {
    //      gathering information about number of VNs in each process within VNG
    MPI_Gather(&n_vn_p, 1, MPI_INT, n_vn_of_ps_in_vng, 1, MPI_INT, 0, vng_comm);

    if (vng_rank == 0) {
        displ_vng_p = new int[vng_size];
        cumulated_sum_shifted(n_vn_of_ps_in_vng, vng_size, displ_vng_p);
    }
}

void gather_full_vn_prod_from_scan_vng(double* const Vn_prod_from_scan_p, const int n_vn_p, double* const Vn_prod_from_scan_vng,
        int* const n_vn_of_ps_in_vng, int* const displ_vng_p, const MPI_Comm &vng_comm) {
    //      gathering global Vn_prod_from_scan for each VNG
    MPI_Gatherv(Vn_prod_from_scan_p, n_vn_p, MPI_DOUBLE, Vn_prod_from_scan_vng, n_vn_of_ps_in_vng, displ_vng_p, MPI_DOUBLE, 0, vng_comm);
}

void reorder_vn_prod_vng(const int vng_size, const int vng_rank, const int n_vn_vng, double* const Vn_prod_vng, 
        double* const Vn_prod_from_scan_vng, int* const N_vn_vng, int* &N_vn_vng_reordered) {
    if (vng_rank == 0) {
        N_vn_vng_reordered = new int[n_vn_vng];

        int target_ix;

        for (int i = 0; i < n_vn_vng; i++) {
            int target_p = i % vng_size;
            int p_offset = (n_vn_vng / vng_size) * target_p + std::min(n_vn_vng % vng_size, target_p);
            target_ix = p_offset + i / vng_size;
            Vn_prod_vng[target_ix] = Vn_prod_from_scan_vng[i];
            N_vn_vng_reordered[target_ix] = N_vn_vng[i];
        }
    }
}

void distribute_reordered_vn_prod_vng(double* const Vn_prod_vng, int* const N_vn_vng_reordered, int* const n_vn_of_ps_in_vng, 
        int* const displ_vng_p, double* const Vn_prod_p, int* const N_vn_p, int n_vn_p, const MPI_Comm &vng_comm) {
    MPI_Scatterv(Vn_prod_vng, n_vn_of_ps_in_vng, displ_vng_p, MPI_DOUBLE, Vn_prod_p, n_vn_p, MPI_DOUBLE, 0, vng_comm);
    MPI_Scatterv(N_vn_vng_reordered, n_vn_of_ps_in_vng, displ_vng_p, MPI_INT, N_vn_p, n_vn_p, MPI_INT, 0, vng_comm);
}

#pragma endregion


#pragma region VNG-masters group

void scatter_vng_sizes(int* const Vng_n_vns, int& n_vn_vng, const MPI_Comm& masters_comm) {
    // inform each master about number of VNs in its VNG

    MPI_Scatter(Vng_n_vns, 1, MPI_INT, &n_vn_vng, 1, MPI_INT, AGDS_MASTER_RANK, masters_comm);
}

void distribute_VN_v_and_VN_n(const int rank, const int n_vn_vng, int* const Vng_n_vns, double* const trees, double* const tree,
        int* const N_vn_vngs, int* const N_vn_vng, const MPI_Comm& masters_comm) {
    // distribute VN_v and VN_n between masters of VNGs

    int displs[N_GROUPS];
    if (rank == AGDS_MASTER_RANK) {
        cumulated_sum_shifted(Vng_n_vns, N_GROUPS, displs);
    }

    MPI_Scatterv(trees, Vng_n_vns, displs, MPI_DOUBLE, tree, n_vn_vng, MPI_DOUBLE, AGDS_MASTER_RANK, masters_comm);
    MPI_Scatterv(N_vn_vngs, Vng_n_vns, displs, MPI_INT, N_vn_vng, n_vn_vng, MPI_INT, AGDS_MASTER_RANK, masters_comm);
}

#pragma endregion


#pragma region World group

// share number of processes belonging to each group (direct and cumulated)
void share_vng_proc_sizes(int* const master_ranks, int* const Vng_n_p) {
    MPI_Bcast(master_ranks, N_GROUPS, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(Vng_n_p, N_GROUPS, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
}

// create group and communicator for masters of VNGs
void setup_vng_communicators(const MPI_Group& world_group, int* const master_ranks, MPI_Group& masters_group, MPI_Comm& masters_comm) {
    MPI_Group_incl(world_group, N_GROUPS, master_ranks, &masters_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, masters_group, MASTERS_TAG, &masters_comm);
}

void broadcast_conn_matrix(int size, int rank, int& n_on_p, int* const CONN, int*& CONN_p) {
    int* CONN_len_all_proc;
    int* displacements;
    if (rank == AGDS_MASTER_RANK) {
        CONN_len_all_proc = new int[size];
        displacements = new int[size];
    }

    n_on_p = N_ON / size + (N_ON % size > rank ? 1 : 0);

    MPI_Gather(&n_on_p, 1, MPI_INT, CONN_len_all_proc, 1, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);

    if (rank == AGDS_MASTER_RANK) {
        for (int i = 0; i < size; i++) {
            CONN_len_all_proc[i] *= N_GROUPS;
        }
        cumulated_sum_shifted(CONN_len_all_proc, size, displacements);
    }

    CONN_p = new int[n_on_p * N_GROUPS];
    MPI_Scatterv(CONN, CONN_len_all_proc, displacements, MPI_INT, CONN_p, n_on_p * N_GROUPS, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);

    if (rank == AGDS_MASTER_RANK) {
        delete[] CONN_len_all_proc;
        delete[] displacements;
    }
}

#pragma endregion


#pragma region World Master-specific

void get_global_vn_conn_indices(int* CONN_global, int* CONN, int* CONN_ix, int* VN_conn_counts_cumulated) {
    for (int i = 0; i < N_ON * N_GROUPS; i++) {
        CONN_global[i] = VN_conn_counts_cumulated[CONN[i]] + CONN_ix[i];
    }
}


int* divide_workers(int* counts, int mpi_size, int* masters_indices, int* Vng_n_p) {
    int n_workers = mpi_size;
    int* worker_division = new int[N_GROUPS];
    int* counts_tmp = new int[N_GROUPS];
    int total_count, assigned_workers, max_elem_ix;
    bool any_worker_assigned;

    for (int i = 0; i < N_GROUPS; i++) {
        worker_division[i] = 0;
        counts_tmp[i] = counts[i];
    }

    while (n_workers > 0) {
        total_count = sum(counts, N_GROUPS);
        any_worker_assigned = false;

        for (int i = 0; i < N_GROUPS; i++) {
            assigned_workers = n_workers * counts_tmp[i] / total_count;
            worker_division[i] += assigned_workers;
            counts_tmp[i] = counts[i] / (1 + worker_division[i]);
            
            any_worker_assigned |= assigned_workers > 0;
        }

        if (!any_worker_assigned) {
            max_elem_ix = max_elem_index(counts_tmp, N_GROUPS);
            worker_division[max_elem_ix] += 1;
            counts_tmp[max_elem_ix] = counts[max_elem_ix] / (1 + worker_division[max_elem_ix]);
        }

        n_workers = mpi_size - sum(worker_division, N_GROUPS);
        any_worker_assigned = false;
    }

    masters_indices[0] = 0;
    for (int g = 1; g < N_GROUPS; g++) {
        masters_indices[g] = masters_indices[g-1] + worker_division[g-1];
    }

    for (int g = 0; g < N_GROUPS; g++) {
        Vng_n_p[g] = worker_division[g];
    }

    int* colours = new int[mpi_size];

    int colour = 0;
    for (int i = 0; i < mpi_size; i++) {
        while (worker_division[colour] == 0) {
            colour++;
        }
        colours[i] = colour;
        worker_division[colour]--;
    }

    delete[] worker_division;
    delete[] counts_tmp;

    return colours;
}


int build_tree(double* values, double* tree, int* counts, int* CONN, int* CONN_ix, int g_ix) {
    // mock implementation
    double* tree_tmp = new double[N_ON];
    int* counts_tmp = new int[N_ON];
    int distinct_count = 0;
    bool found;

    for (int i = 0; i < N_ON; i++) {
        found = false;

        for (int j = 0; j < distinct_count; j++) {
            if (fabs(tree_tmp[j] - values[i]) < EPSILON) {
                counts_tmp[j]++;
                CONN[i * N_GROUPS + g_ix] = j;
                CONN_ix[i * N_GROUPS + g_ix] = counts_tmp[j] - 1;
                found = true;
                break;
            }
        }

        if (!found) {
            tree_tmp[distinct_count] = values[i];
            counts_tmp[distinct_count] = 1;
            CONN[i * N_GROUPS + g_ix] = distinct_count;
            CONN_ix[i * N_GROUPS + g_ix] = 0;
            distinct_count++;
        }
    }

    std::vector<int> indices = sort_indices(tree_tmp, distinct_count);

    int i_dest = 0;
    for (auto i: indices) {
        tree[i_dest] = tree_tmp[i];
        counts[i_dest] = counts_tmp[i];
        i_dest++;
    }

    for (int i = 0; i < N_ON; i++) {
        CONN[i * N_GROUPS + g_ix] = indices[CONN[i * N_GROUPS + g_ix]];
    }

    delete[] tree_tmp;
    delete[] counts_tmp;

    return distinct_count;
}

// init data, divide processes into groups, build mock tree for each VNG
void setup_data_and_groups(int size, double* data, double* trees, int* N_vn_vngs, int* CONN, int* CONN_global_ix,
        int* Vng_n_vns, int* worker_spread, int* master_ranks, int* Vng_n_p) {
    data = init_full_data(N_GROUPS, N_ON);

    int* CONN_local_ix = new int[N_ON * N_GROUPS];

    int offset = 0;
    for (int g = 0; g < N_GROUPS; g++) {
        Vng_n_vns[g] = build_tree(&(data[g * N_ON]), &(trees[offset]), &(N_vn_vngs[offset]), CONN, CONN_local_ix, g);
        offset += Vng_n_vns[g];
        printf("N_vns_vng_%d: %d\n", g, Vng_n_vns[g]);
    }

    worker_spread = divide_workers(Vng_n_vns, size, master_ranks, Vng_n_p);

    // compute vn-on connections global indices
    int* VN_conn_counts_cumulated = new int[N_ON * N_GROUPS];
    cumulated_sum(N_vn_vngs, sum(Vng_n_vns, N_GROUPS), VN_conn_counts_cumulated);

    get_global_vn_conn_indices(CONN_global_ix, CONN, CONN_local_ix, VN_conn_counts_cumulated);

    delete[] CONN_local_ix;
    delete[] VN_conn_counts_cumulated;
}

#pragma endregion


int main(int argc, char** argv)
{
    #pragma region Init MPI

    MPI_Init(&argc, &argv);

    int size, rank;
    mpi_size_and_rank(MPI_COMM_WORLD, size, rank);

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    MPI_Group masters_group;
    MPI_Comm masters_comm;

    MPI_Comm vng_comm;

    #pragma endregion

    #pragma region Init variables

    double* data;
    double* trees = new double[N_GROUPS * N_ON];;
    int* N_vn_vngs = new int[N_GROUPS * N_ON];
    int* Vng_n_vns = new int[N_GROUPS];
    int* worker_spread;
    int master_ranks[N_GROUPS];
    int Vng_n_p[N_GROUPS];

    int* CONN = new int[N_ON * N_GROUPS];
    int* CONN_global_ix = new int[N_ON * N_GROUPS];
    int* CONN_p;

    double *values, *tree;
    int *counts, *N_vn_vng;
    int n_vn_vng;
    int n_on_p;

    int vng_id;
    int vng_size, vng_rank;

    double* Vn_prod_from_scan_p;
    double vn_range;
    int n_vn_p;

    int* displ_vng_p;

    int* n_vn_of_ps_in_vng;
    double* Vn_prod_from_scan_vng;
    double* Vn_prod_vng;
    double* Vn_prod_p;

    int* N_vn_vng_reordered;

    #pragma endregion

    // setup global Master
    if (rank == AGDS_MASTER_RANK) { // process is a global Master
        setup_data_and_groups(size, data, trees, N_vn_vngs, CONN, CONN_global_ix, Vng_n_vns, worker_spread, master_ranks, Vng_n_p);
    }

    // share basic data with everybody
    share_vng_proc_sizes(master_ranks, Vng_n_p);
    setup_vng_communicators(world_group, master_ranks, masters_group, masters_comm);
    broadcast_conn_matrix(size, rank, n_on_p, CONN, CONN_p);
    

    // setup VNG masters
    if (masters_comm != MPI_COMM_NULL) { // process is a Master of one of the VNGs
        int masters_size, masters_rank;
        mpi_size_and_rank(masters_comm, masters_size, masters_rank);

        scatter_vng_sizes(Vng_n_vns, n_vn_vng, masters_comm);

        tree = new double[n_vn_vng];
        N_vn_vng = new int[n_vn_vng];

        distribute_VN_v_and_VN_n(rank, n_vn_vng, Vng_n_vns, trees, tree, N_vn_vngs, N_vn_vng, masters_comm);
    }

    // setup VNGs
    split_processes_into_vngs(worker_spread, rank, vng_id, vng_size, vng_rank, vng_comm);
    broadcast_number_of_vns_in_vngs(n_vn_vng, vng_comm);
    // debug_printf_from_vng_leaders(rank, vng_rank, vng_id, n_vn_vng);
    compute_vn_prod(tree, n_vn_vng, Vn_prod_from_scan_p, vn_range, n_vn_p, vng_comm);
    // debug_printf_from_all_workers(rank, vng_id, n_vn_p);
    broadcast_number_of_vns_in_vngs(Vng_n_vns);

    if (vng_rank == 0) {
        n_vn_of_ps_in_vng = new int[vng_size];
        Vn_prod_vng = new double[n_vn_vng];
        Vn_prod_from_scan_vng = new double[n_vn_vng];
    }
    Vn_prod_p = new double[n_vn_p];

    gather_number_of_vns_per_proc(n_vn_p, n_vn_of_ps_in_vng, displ_vng_p, vng_rank, vng_size, vng_comm);
    gather_full_vn_prod_from_scan_vng(Vn_prod_from_scan_p, n_vn_p, Vn_prod_from_scan_vng, n_vn_of_ps_in_vng, displ_vng_p, vng_comm);
    reorder_vn_prod_vng(vng_size, vng_rank, n_vn_vng, Vn_prod_vng, Vn_prod_from_scan_vng, N_vn_vng, N_vn_vng_reordered);
    int N_vn_p[n_vn_p];
    distribute_reordered_vn_prod_vng(Vn_prod_vng, N_vn_vng_reordered, n_vn_of_ps_in_vng, displ_vng_p, Vn_prod_p, N_vn_p, n_vn_p, vng_comm);


    // PREDICT
    // compute inferences

    setup_for_inference(n_vn_p, n_on_p, N_GROUPS, Vng_n_p, NULL, n_vn_vng, vng_id, vng_comm, CONN_p, Vn_prod_p, N_vn_p,
        0, 0, 0, NULL, NULL, NULL, NULL);

    int* all_activated_vns = new int[ACTIVATED_VNS_PER_GROUP * N_GROUPS];
    int* on_queries = new int[ACTIVATED_ONS * N_QUERIES];

    int n_activated_vns, n_activated_ons;

    double start_time, end_time;

    if (K == 1) {
        if (rank == AGDS_MASTER_RANK) {
            srand(SEED);
            mock_on_queries(on_queries, N_QUERIES, N_ON, ACTIVATED_ONS);
        }

        MPI_Bcast(on_queries, ACTIVATED_ONS * N_QUERIES, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);

        int local_on_queries[ACTIVATED_ONS * N_QUERIES];
        int local_on_queries_lengths[N_QUERIES];
        int local_on_queries_displacements[N_QUERIES];
        int offset = 0;

        for (int q = 0; q < N_QUERIES; q++) {
            local_on_queries_lengths[q] = 0;
            local_on_queries_displacements[q] = offset;
            for (int i = 0; i < ACTIVATED_ONS; i++) {
                if (on_queries[q * ACTIVATED_ONS + i] % size == rank) {
                    local_on_queries[offset] = on_queries[q * ACTIVATED_ONS + i];
                    local_on_queries_lengths[q]++;
                    offset++;
                }
            }
        }

        // logging init
        MPE_Init_log();
        init_events();

        // ********************************** MEASUREMENTS **********************************

        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        for (int q_ix = 0; q_ix < N_QUERIES; q_ix++) {
            inference(NULL, 0, &(local_on_queries[local_on_queries_displacements[q_ix]]), local_on_queries_lengths[q_ix], false, K);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        end_time = MPI_Wtime();

        // ********************************** MEASUREMENTS END **********************************
        
        const int log_file_max_length = 50;
        char log_file_name[log_file_max_length];
        snprintf(log_file_name, log_file_max_length, "inference%d", size);

        MPE_Finish_log(log_file_name);
    }
    else {
        // generate random inactive (output) VNGs in each query, done at global master
        int vn_queries_inactive_vngs[N_QUERIES * (N_GROUPS - ACTIVATED_VNGS)];

        if (rank == AGDS_MASTER_RANK) {
            srand(SEED);

            mock_vn_queries_inactive_vngs(vn_queries_inactive_vngs, N_QUERIES, N_GROUPS, ACTIVATED_VNGS);
        }

        // generate random VNs for queries, done at group masters
        int vn_queries_vng[N_QUERIES * ACTIVATED_VNS_PER_GROUP];
        if (vng_rank == 0) {
            MPI_Bcast(vn_queries_inactive_vngs, N_QUERIES * (N_GROUPS - ACTIVATED_VNGS), MPI_INT, AGDS_MASTER_RANK, masters_comm);
            
            mock_vn_queries_vng(vn_queries_inactive_vngs, vn_queries_vng, N_GROUPS, N_QUERIES, ACTIVATED_VNS_PER_GROUP);
        }

        MPI_Bcast(vn_queries_vng, N_QUERIES * ACTIVATED_VNS_PER_GROUP, MPI_INT, 0, vng_comm);

        // find local VNs in queries
        int local_vn_queries[N_QUERIES * ACTIVATED_VNS_PER_GROUP];
        int local_vn_queries_lengths[N_QUERIES];
        int local_vn_queries_displs[N_QUERIES];
        bool local_vn_queries_active[N_QUERIES];
        int offset = 0;

        for (int q = 0; q < N_QUERIES; q++) {
            local_vn_queries_lengths[q] = 0;
            local_vn_queries_displs[q] = offset;

            if (vn_queries_inactive_vngs[2*q] == vng_id || vn_queries_inactive_vngs[2*q + 1] == vng_id) {
                local_vn_queries_active[q] = false;
            }
            else {
                local_vn_queries_active[q] = true;
                for (int i = 0; i < ACTIVATED_VNS_PER_GROUP; i++) {
                    if (on_queries[q * ACTIVATED_VNS_PER_GROUP + i] % vng_size == vng_rank) {
                        local_vn_queries[offset] = vn_queries_vng[q * ACTIVATED_ONS + i];
                        local_vn_queries_lengths[q]++;
                        offset++;
                    }
                }
            }
        }

        // logging init
        MPE_Init_log();
        init_events();

        // ********************************** MEASUREMENTS **********************************

        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        for (int q_ix = 0; q_ix < N_QUERIES; q_ix++) {
            inference(&(local_vn_queries[local_vn_queries_displs[q_ix]]), local_vn_queries_lengths[q_ix], NULL, 0, local_vn_queries_active[q_ix], K);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        end_time = MPI_Wtime();

        // ********************************** MEASUREMENTS END **********************************
        
        const int log_file_max_length = 50;
        char log_file_name[log_file_max_length];
        snprintf(log_file_name, log_file_max_length, "inference%d", size);

        MPE_Finish_log(log_file_name);
    }

    teardown_inference();

    // REPORTING

    if (rank == AGDS_MASTER_RANK) {
        printf("Time: %.2fs\n", end_time - start_time);
    }

    
    # pragma region Cleanup

    delete[] Vng_n_vns;
    delete[] CONN_p;
    delete[] Vn_prod_p;
    delete[] Vn_prod_from_scan_p;

    if (rank == AGDS_MASTER_RANK) {
        delete[] data;

        delete[] trees;
        delete[] N_vn_vngs;

        delete[] CONN;
        delete[] CONN_global_ix;

        delete[] worker_spread;
    //     // delete[] query;
    }

    MPI_Group_free(&world_group);
    MPI_Group_free(&masters_group);
    
    if (masters_comm != MPI_COMM_NULL) {
        delete[] tree;
        delete[] N_vn_vng;
        delete[] n_vn_of_ps_in_vng;
        delete[] Vn_prod_vng;
        delete[] Vn_prod_from_scan_vng;
        delete[] displ_vng_p;
        delete[] N_vn_vng_reordered;
        MPI_Comm_free(&masters_comm);
    }

    MPI_Comm_free(&vng_comm);

    MPI_Finalize();

    #pragma endregion


    return 0;
}