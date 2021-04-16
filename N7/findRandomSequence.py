import time
import numpy as np
import random
from MOCUProposed import *

def findRandomSequence(save_f_inv, D_save, init_MOCU_val, K_max, w, N, h , M, T,
                  a_lower_bound_update, a_upper_bound_update, it_idx, update_cnt):
    optimalExperiments = []
    #aaa = np.zeros((N,N))
    timeComplexity = np.ones(update_cnt)
    Rand_seq = np.ones(update_cnt+1)*50.0     
    Rand_seq[0] = init_MOCU_val

    i_set = np.zeros(update_cnt)
    j_set = np.zeros(update_cnt)
    ind_list = []
    for i in range(N):
        for j in range(i + 1, N):
            ind_list.append([[i, j]])
    random.shuffle(ind_list)

    for i in range(update_cnt):
        iterationStartTime = time.time()
        i_set[i] = np.asarray(ind_list[i])[0][0]
        j_set[i] = np.asarray(ind_list[i])[0][1]
        iterationTime = time.time() - iterationStartTime
        timeComplexity[i] = iterationTime

    for ij in range(1,update_cnt+1):
        iterationStartTime = time.time()

        i = int(i_set[ij-1])
        j = int(j_set[ij-1])

        #print(i,j)

        optimalExperiments.append((i, j))
        f_inv = save_f_inv[i, j]

        if D_save[i, j] == 0.0:
            a_upper_bound_update[i, j] \
                = min(a_upper_bound_update[i, j], f_inv)
            a_upper_bound_update[j, i] \
                = min(a_upper_bound_update[i, j], f_inv)
        else:
            a_lower_bound_update[i, j] \
                = max(a_lower_bound_update[i, j], f_inv)
            a_lower_bound_update[j, i] \
                = max(a_lower_bound_update[i, j], f_inv)

        print("Iteration: ", ij, ", selected: (", i, j, ")", time.time() - iterationStartTime, "seconds")
        it_temp_val = np.zeros(it_idx)
        for l in range(it_idx):
            # it_temp_val[l] = MOCU(K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update, ((ij * N * N * N) + l))
            it_temp_val[l] = MOCU(K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update, 0)
        Rand_seq[ij] = np.mean(it_temp_val)  
        print("before adjusting")
        print(Rand_seq[ij])
        if Rand_seq[ij] > Rand_seq[ij - 1]:
            Rand_seq[ij] = Rand_seq[ij - 1]
        print("The end of iteration: actual MOCU", Rand_seq[ij])
    print(optimalExperiments)
    return Rand_seq, optimalExperiments, timeComplexity

