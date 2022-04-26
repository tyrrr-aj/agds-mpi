#ifndef MOCK_AGDS_DATA_H
#define MOCK_AGDS_DATA_H

// mock data

double* init_full_data(int n_groups, int n_on);


// mock queries

void mock_on_queries(int* activated_ons, int n_queries, int n_on, int n_activated_ons);
void mock_vn_queries_inactive_vngs(int* vn_queries_inactive_vngs, int n_queries, int n_groups, int n_activated_vngs);
void mock_vn_queries_vng(int* vn_queries_inactive_vngs, int* vn_queries_vng, int n_groups, int n_queries, int n_activated_vns_per_group);


#endif