#ifndef INFERENCE
#define INFERENCE

void setup_for_inference(int n_vn_p, int n_on_p, int n_groups, int* vng_sizes, int n_vn_vng, int vng_ix, MPI_Comm vng_comm, int* conn, double* Vn_r, int* Vn_n);
void teardown_inference();

void inference(int* activated_vns, int n_activated_vns, int* activated_ons, int n_activated_ons, bool vng_in_query, int steps);

#endif