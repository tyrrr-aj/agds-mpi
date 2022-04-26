#include <random>

#include "mock_agds_data.hpp"


// mock data

double* init_full_data(int n_groups, int n_on) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double* data = new double[n_groups * n_on];
    for (int i = 0; i < n_groups; i++) {
        for (int j = 0; j < n_on; j++) {
            data[i * n_on + j] = dis(gen);
        }
    }

    return data;
}


// mock queries

void mock_on_queries(int* activated_ons, int n_queries, int n_on, int n_activated_ons) {
    bool unique_confirmed, duplicate_confirmed;
    int candidate;

    for (int q = 0; q < n_queries; q++) {
        for (int i = 0; i < n_activated_ons; i++) {
            unique_confirmed = false;

            while (!unique_confirmed) {
                candidate = rand() % n_on;
                duplicate_confirmed = false;

                for (int j = 0; j < i; j++) {
                    if (activated_ons[q * n_activated_ons + j] == candidate) {
                        duplicate_confirmed = true;
                        break;
                    }
                }

                unique_confirmed = !duplicate_confirmed;
            }

            activated_ons[q * n_activated_ons + i] = candidate;
        }
    }
}


void mock_vn_queries_inactive_vngs(int* vn_queries_inactive_vngs, int n_queries, int n_groups, int n_activated_vngs) {
    bool unique_confirmed, duplicate_confirmed;
    int candidate;

    for (int q = 0; q < n_queries; q++) {
        for (int i = 0; i < n_groups - n_activated_vngs; i++) {
            unique_confirmed = false;

            while (!unique_confirmed) {
                candidate = rand() % n_groups;
                duplicate_confirmed = false;

                for (int j = 0; j < i; j++) {
                    if (vn_queries_inactive_vngs[q * (n_groups - n_activated_vngs) + j] == candidate) {
                        duplicate_confirmed = true;
                        break;
                    }
                }

                unique_confirmed = !duplicate_confirmed;
            }

            vn_queries_inactive_vngs[q * (n_groups - n_activated_vngs) + i] = candidate;
        }
    }
}


void mock_vn_queries_vng(int* vn_queries_inactive_vngs, int* vn_queries_vng, int n_groups, int n_queries, int n_activated_vns_per_group) {
    bool unique_confirmed, duplicate_confirmed;
    int candidate;

    for (int q = 0; q < n_queries; q++) {
        if (!vn_queries_inactive_vngs[2*q] && !vn_queries_inactive_vngs[2*q + 1]) {
            for (int i = 0; i < n_activated_vns_per_group; i++) {
                unique_confirmed = false;

                while (!unique_confirmed) {
                    candidate = rand() % n_groups;
                    duplicate_confirmed = false;

                    for (int j = 0; j < i; j++) {
                        if (vn_queries_inactive_vngs[q * n_activated_vns_per_group + j] == candidate) {
                            duplicate_confirmed = true;
                            break;
                        }
                    }

                    unique_confirmed = !duplicate_confirmed;
                }

                vn_queries_inactive_vngs[q * n_activated_vns_per_group + i] = candidate;
            }
        }
    }
}
