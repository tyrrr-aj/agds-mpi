#ifndef INFERENCE
#define INFERENCE

void setup_for_inference(int n_on, int n_vn_p, int n_on_p, int n_groups, int* vng_sizes_proc, int* vng_sizes_vns, int n_vn_vng, int vng_ix, 
                        MPI_Comm vng_comm, int* conn, int* conn_global_ix, double* Vn_r, int* Vn_n, int n_conn_global, int n_conn_proc, 
                        int n_vn_conn_proc, int* vns_n_conns, int* vns_distribution, int* vns_proc, int* vns_proc_on_to_vn);

void teardown_inference();

void inference(int* activated_vns, int n_activated_vns, int* activated_ons, int n_activated_ons, bool vng_in_query, int steps);

#endif