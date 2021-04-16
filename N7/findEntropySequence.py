import numpy as np
import time
from MOCUProposed import *

def findEntropySequence(save_f_inv, D_save, init_MOCU_val, K_max, w, N, h , M, T,
                  a_lower_bound_update, a_upper_bound_update, it_idx, update_cnt):

    a_diff = np.zeros((N,N))

    Entropy_seq = np.ones(update_cnt+1)*50.0
    Entropy_seq[0] = init_MOCU_val

    timeComplexity = np.ones(update_cnt)

    a_diff = np.triu(a_upper_bound_update - a_lower_bound_update, 1)
    optimalExperiments = []
    #print(a_diff)
    for ij in range(1,update_cnt+1):
        iterationStartTime = time.time()
        flag = 0

        max_ind = np.where(a_diff == np.max(a_diff[np.nonzero(a_diff)]))
        if len(max_ind[0]) == 1:
            i = int(max_ind[0])
            j = int(max_ind[1])
        else:
            i = int(max_ind[0][0])
            j = int(max_ind[1][0])
        
        
        iterationTime = time.time() - iterationStartTime
        timeComplexity[ij-1] = iterationTime

        optimalExperiments.append((i, j))
        a_diff[i, j] = 0.0

        #print(i,j)

        f_inv = save_f_inv[i, j]

        if D_save[i, j] == 0.0:
            a_upper_bound_update[i, j] \
                = min(a_upper_bound_update[i, j], f_inv)
            a_upper_bound_update[j, i] \
                = min(a_upper_bound_update[i, j], f_inv)
            if f_inv > a_upper_bound_update[i, j]:
                flag = 1

        else:
            a_lower_bound_update[i, j] \
                = max(a_lower_bound_update[i, j], f_inv)
            a_lower_bound_update[j, i] \
                = max(a_lower_bound_update[i, j], f_inv)
            if f_inv < a_lower_bound_update[i, j]:
                flag = 1
        
        print("Iteration: ", ij, ", selected: (", i, j, ")", iterationTime, "seconds")
        it_temp_val = np.zeros(it_idx)
        for l in range(it_idx):
            it_temp_val[l] = MOCU(K_max, w, N, h , M, T, a_lower_bound_update, a_upper_bound_update, 0)
        Entropy_seq[ij] = np.mean(it_temp_val)  
        print("before adjusting")
        print(Entropy_seq[ij])
        if Entropy_seq[ij] > Entropy_seq[ij - 1]:
            Entropy_seq[ij] = Entropy_seq[ij - 1]
        print("The end of iteration: actual MOCU", Entropy_seq[ij])
    print(optimalExperiments)
    return Entropy_seq, optimalExperiments, timeComplexity