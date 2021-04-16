import time
import numpy as np
from MOCUProposed import *

def findIdealequence(syncThresholds, isSynchronized, MOCUInitial, K_max, w, N, h , MVirtual, MReal, TVirtual, TReal, aLowerBoundIn, aUpperBoundIn, it_idx, update_cnt, proposed, iterative = True):

    pseudoRandomSequence = True

    MOCUCurve = np.ones(update_cnt+1)*50.0
    MOCUCurve[0] = MOCUInitial
    timeComplexity = np.ones(update_cnt)

    aUpperBoundUpdated = aUpperBoundIn.copy()
    aLowerBoundUpdated = aLowerBoundIn.copy()

    optimalExperiments  = []

    for iteration in range(1, update_cnt+1):
        R = np.zeros((N,N))
        iterationStartTime = time.time()
        for i in range(N):
            for j in range(i+1,N):
                if (i, j) not in optimalExperiments:
                    aUpper = aUpperBoundUpdated.copy()
                    aLower = aLowerBoundUpdated.copy()

                    f_inv = syncThresholds[i, j]
            
                    if isSynchronized[i, j] == 0.0:
                        aUpper[i, j] = min(aUpperBoundUpdated[i, j], f_inv)
                        aUpper[j, i] = min(aUpperBoundUpdated[i, j], f_inv)
                    else:
                        aLower[i, j] = max(aLowerBoundUpdated[i, j], f_inv)
                        aLower[j, i] = max(aLowerBoundUpdated[i, j], f_inv)

                    it_temp_val = np.zeros(it_idx)
                    for l in range(it_idx):
                        it_temp_val[l] = MOCU(K_max, w, N, h, MReal, TReal, aLower, aUpper, 0)
                    R[i, j] = np.mean(it_temp_val)  

        min_ind = np.where(R == np.min(R[np.nonzero(R)]))

        if len(min_ind[0]) == 1:
            min_i_MOCU = int(min_ind[0])
            min_j_MOCU = int(min_ind[1])
        else:
            min_i_MOCU = int(min_ind[0][0])
            min_j_MOCU = int(min_ind[1][0])

        iterationTime = time.time() - iterationStartTime
        timeComplexity[iteration - 1] = iterationTime

        optimalExperiments.append((min_i_MOCU, min_j_MOCU))

        MOCUVal = R[min_i_MOCU, min_j_MOCU]
        R[min_i_MOCU, min_j_MOCU] = 0.0

        f_inv = syncThresholds[min_i_MOCU, min_j_MOCU]
        
        if isSynchronized[min_i_MOCU, min_j_MOCU] == 0.0:
            aUpperBoundUpdated[min_i_MOCU, min_j_MOCU] \
                = min(aUpperBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
            aUpperBoundUpdated[min_j_MOCU, min_i_MOCU] \
                = min(aUpperBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
        else:
            aLowerBoundUpdated[min_i_MOCU, min_j_MOCU] \
                = max(aLowerBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)
            aLowerBoundUpdated[min_j_MOCU, min_i_MOCU] \
                = max(aLowerBoundUpdated[min_i_MOCU, min_j_MOCU], f_inv)

        print("Iteration: ", iteration, ", selected: (", min_i_MOCU, min_j_MOCU, ")", iterationTime, "seconds")
        MOCUCurve[iteration] = MOCUVal
        print("before adjusting")
        print(MOCUCurve[iteration])
        if MOCUCurve[iteration] > MOCUCurve[iteration - 1]:
            MOCUCurve[iteration] = MOCUCurve[iteration - 1]
        print("The end of iteration: actual MOCU", MOCUCurve[iteration])
    print(optimalExperiments)
    return MOCUCurve, optimalExperiments, timeComplexity