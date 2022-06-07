#include <random>

#include "mock_agds_data.hpp"


// mock data

// random
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

// prepared
// double* init_full_data(int n_groups, int n_on) {
//     if (n_groups != 2 || n_on != 10) {
//         printf("Error: using prepared mock data with unmatching N_GROUPS and N_ON\n");
//         exit(1);
//     }

//     // return new double[n_groups * n_on] {1.0, 1.0, 7.0, 1.0, 1.0, 2.0, 7.0, 2.0, 1.0, 2.0, 7.0, 1.0, 7.0, 7.0, 8.0, 7.0, 3.0, 11.0, 11.0, 11.0};
//     return new double[n_groups * n_on] {1.0, 7.0, 1.0, 7.0, 1.0, 7.0, 7.0, 8.0, 3.0, 11.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 7.0, 7.0, 11.0, 11.0};
// }


// mock queries

// random
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

// prepared
// void mock_on_queries(int* activated_ons, int n_queries, int n_on, int n_activated_ons) {
//     if (n_queries != 2 || n_on != 10 || n_activated_ons != 5) {
//         printf("Error: using prepared mock data with unmatching N_QUERIES, N_ACTIVATED_ONS and N_ON\n");
//         exit(1);
//     }

//     activated_ons[0] = 0;
//     activated_ons[1] = 3;
//     activated_ons[2] = 6;
//     activated_ons[3] = 7;
//     activated_ons[4] = 9;
//     activated_ons[5] = 0;
//     activated_ons[6] = 3;
//     activated_ons[7] = 6;
//     activated_ons[8] = 7;
//     activated_ons[9] = 9;
// }


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
