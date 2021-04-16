import time
import pycuda.autoinit

import pycuda.driver as drv
import pandas as pd
import numpy as np

from pycuda.compiler import SourceModule

mod = SourceModule("""

// This should be manually changed due to the technical issue in the PyCUDA.
// Well, yes, I am lazy...
#include <stdio.h>

#define N_global 8
#define NUMBER_FEATURES (N_global * N_global)

__device__ int mocuCompByNN(double *w, int N, double* a, double *l1w, double *l1b, double *l2w, double *l2b, int equatorIndex, double *wOrder)
{
    int D = 0;
    int i, j;

    int fillingIndex = 0;
    double x[NUMBER_FEATURES];

    for (i = 0; i < N_global; i++) {
        x[i] = w[i];
        fillingIndex++;
    }

    int aIndex = 0;
    for (i = 0; i < N_global; i++) {
        for (j = i + 1; j < N_global; j++) {
            x[fillingIndex] = w[i] - w[j];
            fillingIndex++;
            x[fillingIndex] = a[aIndex];
            fillingIndex++;
            aIndex++;
        }
    }

    double h1o[NUMBER_FEATURES*4];
    for (i = 0; i < NUMBER_FEATURES*4; i++) {
        h1o[i] = 0;
        for (j = 0; j < NUMBER_FEATURES; j++) {
             h1o[i] += (x[j]*l1w[(j*NUMBER_FEATURES*4) + i]); 
        }
        h1o[i] = max(h1o[i] + l1b[i], 0.0);
    }

    double h2o[2];
    for (i = 0; i < 2; i++) {
        h2o[i] = 0;
        for (j = 0; j < NUMBER_FEATURES*4; j++) {
             h2o[i] += (h1o[j]*l2w[(j*2) + i]);
        }
        h2o[i] = h2o[i] + l2b[i];
    }

    double f[2];
    for (i = 0; i < 2; i++) {
        f[i] = exp(h2o[i] - max(h2o[0], h2o[1]));
    }

    if ( (f[0]/(f[0] + f[1])) < (f[1]/(f[0] + f[1])) ) {
        D = 1;
    }
    else {
        D = 0;
    }
   
    return D;
}

__global__ void taskNN(double *l1w, double *l1b, double *l2w, double *l2b, int equatorIndex, double *wOrder, double *random_data, double *a_save, double *w, \
                     double h , int N, int M, double *a_lower_bound_update, double *a_upper_bound_update, int pseudoRandomSequence, double *aOrder)
{
    const int i_c = blockDim.x*blockIdx.x + threadIdx.x;
    int i;
    int observeIndex = 100000000000;
    if (i_c == observeIndex) {
        printf("l1w\\n");
        for (i=0; i<N_global; i++){
            printf("%0.10f\\t", l1w[i]);
        }
        printf("\\n");

        printf("l1b\\n");
        for (i=0; i<N_global; i++){
            printf("%0.10f\\t", l1b[i]);
        }
        printf("\\n");

        printf("l2w\\n");
        for (i=0; i<N_global; i++){
            printf("%0.10f\\t", l2w[i]);
        }
        printf("\\n");

        printf("l2b\\n");
        for (i=0; i<2; i++){
            printf("%0.10f\\t", l2b[i]);
        }
        printf("\\n");

        printf("wOrder\\n");
        for (i=0; i<N_global; i++){
            printf("%f\\t", wOrder[i]);
        }
        printf("\\n");

        printf("random_data\\n");
        for (i=0; i<N_global; i++){
            printf("%0.10f\\t", random_data[i]);
        }
        printf("\\n");

        printf("w\\n");
        for (i=0; i<N_global; i++){
            printf("%0.10f\\t", w[i]);
        }
        printf("\\n");

        printf("a_lower_bound_update\\n");
        for (i=0; i<N_global; i++){
            printf("%0.10f\\t", a_lower_bound_update[i]);
        }
        printf("\\n");

        printf("a_upper_bound_update\\n");
        for (i=0; i<N_global; i++){
            printf("%0.10f\\t", a_upper_bound_update[i]);
        }
        printf("\\n");

        printf("aOrder\\n");
        for (i=0; i<N_global; i++){
            printf("%f\\t", aOrder[i]);
        }
        printf("\\n");

        printf("frequence\\n");
        for (i=0; i<N_global; i++){
            printf("%0.10f\\t", w[i]);
        }
        printf("\\n");
    }

    if (i_c == observeIndex) {
        printf("frequence\\n");
        for (i=0; i<N_global; i++){
            printf("%0.10f\\t", w[i]);
        }
        printf("\\n");
    }

    double a_new[N_global*(N_global-1)/2];
    for (i=0; i<N_global*(N_global-1)/2; i++){
        a_new[i] = 0.0;
    }

    int rand_ind, cnt0;
    if (i_c == observeIndex) {
        printf("find minimum cost %d", i_c);
            for (i=0;i<N_global*N_global;i++){
            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
    cnt0 = (i_c*(N-1)*N/2);

    for (i = 0; i < N_global*(N_global-1)/2; i++) {
        rand_ind = cnt0 + aOrder[i];
        if (aOrder[i] != -1) {
            a_new[i] = a_lower_bound_update[i]+ (a_upper_bound_update[i]-a_lower_bound_update[i]) * random_data[rand_ind];
        }
    }
    if (i_c == observeIndex) {
        printf("Initialization of a_new", i_c);
            for (i=0;i<N_global*N_global;i++){
                if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
    bool isFound = 0;
    int D;
    int iteration;
    double initialC = 0;

    for (iteration = 1; iteration < 100; iteration++) {
        initialC = 2 * iteration;
        for (i = 0; i < N_global*(N_global-1)/2; i++) {
            if (aOrder[i] == -1) {
                a_new[i] = initialC;
            }
        }
        if (i_c == observeIndex) {
            printf("Find upper bound, iteration: %d, upperbound: %.10f", iteration, initialC);
            for (i=0;i<N_global*N_global;i++){
                            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
            }
            printf("\\n");
        }
        D = mocuCompByNN(w, N+1, a_new, l1w, l1b, l2w, l2b, equatorIndex, wOrder);
        if (D > 0) {
            isFound = 1;
            break;
        }
    }

    double c_lower = 0.0;
    double c_upper = initialC;
    double midPoint = 0;
    int iterationOffset = iteration - 1;

    if (isFound > 0) {
        for (iteration = 0; iteration < (14 + iterationOffset); iteration++) {
            midPoint = (c_upper + c_lower) / 2.0;
            for (i = 0; i < N_global*(N_global-1)/2; i++) {
                if (aOrder[i] == -1) {
                    a_new[i] = midPoint;
                }
            }
            if (i_c == observeIndex) {
            printf("binary serach, iteration: %d, upper bound: %.10f, lower bound: %.10f", iteration, c_upper, c_lower);
                for (i=0;i<N_global*N_global;i++){
                    if ((i%N_global) == 0) {
                        printf("\\n");
                    }
                    printf("a_new[%d]=%.10f\\t", i, a_new[i]);
                }
                printf("\\n");
            }
            D = mocuCompByNN(w, N+1, a_new, l1w, l1b, l2w, l2b, equatorIndex, wOrder);

            if (D > 0) {  
                c_upper = midPoint;
            }
            else {  
                c_lower = midPoint;
            }

            if ((c_upper - c_lower) < 0.00025) {
                // printf("Upper - Lower is less than 0.00025\\n");
                break;
            }
        }
        a_save[i_c] = c_upper; 
    }
    else {
        printf("Can't find a! i_c: %d\\n", i_c);
        a_save[i_c] = -1; 
    }
    if (i_c == observeIndex) {
        printf("binary serach end, iteration: %d, upper bound: %.10f, lower bound: %.10f", iteration, c_upper, c_lower);
        for (i=0;i<N_global*N_global;i++){
            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
}

__device__ int mocu_comp(double *w, double h, int N, int M, double* a)
{
    int D = 0;
    double tol,max_temp,min_temp;
    max_temp = -100.0;
    min_temp = 100.0;
    double pi_n = 3.14159265358979323846;

    double theta[N_global];
    double theta_old[N_global];
    double F[N_global],k1[N_global],k2[N_global],k3[N_global],k4[N_global];
    double diff_t[N_global];
    int i,j,k;
    double t = 0.0;
    double sum_temp;


    for (i=0;i<N;i++){
        theta[i] = 0.0;
        theta_old[i] = 0.0;
        F[i] = 0.0;
        k1[i] = 0.0;
        k2[i] = 0.0;
        k3[i] = 0.0;
        k4[i] = 0.0;
        diff_t[i] = 0.0;
    }

    for (k=0;k<M;k++){

        
        for (i=0;i<N;i++){

            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
              
            }
            F[i] = w[i] + sum_temp;
        }

        for(i=0;i<N;i++){
            k1[i] = h*F[i];
            theta[i] = theta_old[i] + k1[i]/2.0;
          }
          
        

        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
        }

        for(i=0;i<N;i++){
            k2[i] = h*F[i];
            theta[i] = theta_old[i] + k2[i]/2.0;
          }
          

        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
         }
        for(i=0;i<N;i++){
            k3[i] = h*F[i];
            theta[i] = theta_old[i] + k3[i];
          }



        for (i=0;i<N;i++){
            sum_temp = 0.0;
            for (j=0;j<N;j++){
              sum_temp += a[j*N+i]*sin(theta[j] - theta[i]);
            }
            F[i] = w[i] + sum_temp;
        }


        for(i=0;i<N;i++){        
            k4[i] = h*F[i];
            theta[i] = theta_old[i] + 1.0/6.0*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
          }


        for (i=0;i<N;i++){
            if ((M/2) < k)
            {
             diff_t[i] = (theta[i] - theta_old[i]);
            }

             if (theta[i] > 2.0*pi_n)
             {
          		theta[i] = theta[i] - 2.0*pi_n;
            }

             theta_old[i] = theta[i];  
        }

        if ((M/2) < k){
            for(i=0;i<N;i++){
                if (diff_t[i] > max_temp)
                {
                    max_temp  = diff_t[i];
                }

                if (diff_t[i] < min_temp)
                {
                    min_temp  = diff_t[i];
                }
            }

        }
      
        t = t+h;
      
    }

    
    tol = max_temp-min_temp;
    if (tol <= 0.001){
        D = 1;
    }
    
    return D;
}

__global__ void task(double *a, double *random_data, double *a_save, double *w, \
                     double h , int N, int M, double *a_lower_bound_update, \
                    double *a_upper_bound_update)
{
    const int i_c = blockDim.x*blockIdx.x + threadIdx.x;
    int i,j;
    int observeIndex = 10000000000;
    
    double a_new[N_global*N_global];
    for (i=0;i<N_global*N_global;i++){
            a_new[i] = 0.0;
    }
    if (i_c == observeIndex) {
        printf("find minimum cost %d", i_c);
            for (i=0;i<N_global*N_global;i++){
            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
    int rand_ind, cnt0, cnt1;
    
    cnt0 = (i_c*(N-1)*N/2);
    cnt1 = 0;

    for (i=0;i<N;i++){
        for (j=i+1;j<N;j++)
        {
            rand_ind = cnt0 + cnt1;
            a_new[j*(N+1)+i] = a_lower_bound_update[(j*N)+i]+ (a_upper_bound_update[(j*N)+i]-a_lower_bound_update[(j*N)+i])*random_data[rand_ind];
            a_new[i*(N+1)+j] = a_new[j*(N+1)+i];
            cnt1++;
        }
    }

    if (i_c == observeIndex) {
        printf("Initialization of a_new", i_c);
            for (i=0;i<N_global*N_global;i++){
                            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
    bool isFound = 0;
    int D;
    int iteration;
    double initialC = 0;

    for (iteration = 1; iteration < 100; iteration++) {
        initialC = 2 * iteration;
        for (i=0;i<N;i++){
            a_new[(i*(N+1))+N] = initialC;
            a_new[(N*(N+1))+i] = initialC;
        }

        if (i_c == observeIndex) {
        printf("Find upper bound, iteration: %d, upperbound: %.10f", iteration, initialC);
            for (i=0;i<N_global*N_global;i++){
                            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
            }
            printf("\\n");
        }
        D = mocu_comp(w, h, N+1, M, a_new);

        if (D > 0) {
            isFound = 1;
            break;
        }
    }

    double c_lower = 0.0;
    double c_upper = initialC;
    double midPoint = 0;
    int iterationOffset = iteration - 1;

    if (isFound > 0) {
        for (iteration = 0; iteration < (14 + iterationOffset); iteration++) {
            midPoint = (c_upper + c_lower) / 2.0;

            for (i=0;i<N;i++){
                a_new[(i*(N+1))+N] = midPoint;
                a_new[(N*(N+1))+i] = midPoint;
            }
            if (i_c == observeIndex) {
            printf("binary serach, iteration: %d, upper bound: %.10f, lower bound: %.10f", iteration, c_upper, c_lower);
                for (i=0;i<N_global*N_global;i++){
                    if ((i%N_global) == 0) {
                        printf("\\n");
                    }
                    printf("a_new[%d]=%.10f\\t", i, a_new[i]);
                }
                printf("\\n");
            }
            D = mocu_comp(w, h, N+1, M, a_new);

            if (D > 0) {  
                c_upper = midPoint;
            }
            else {  
                c_lower = midPoint;
            }

            if ((c_upper - c_lower) < 0.00025) {
                //printf("Upper - Lower is less than 0.00025\\n");
                break;
            }
        }
        a_save[i_c] = c_upper; 
    }
    else {
        printf("Can't find a! i_c: %d\\n", i_c);
        a_save[i_c] = -1; 
    }    
    if (i_c == observeIndex) {
        printf("binary serach end, iteration: %d, upper bound: %.10f, lower bound: %.10f", iteration, c_upper, c_lower);
        for (i=0;i<N_global*N_global;i++){
            if ((i%N_global) == 0) {
                printf("\\n");
            }
            printf("a_new[%d]=%.10f\\t", i, a_new[i]);
        }
        printf("\\n");
    }
}

"""
)

task = mod.get_function("task")
taskNN = mod.get_function("taskNN")

def MOCUProposed(K_max, w, N, h , M, T, aLowerBountIn, aUppwerBountIn, seed, pseudoRandomSequence):
    # seed = 0
    # load model
    modelPath = '../models/model_N8_specificSamples50000n50000_timeFrame400_randomInitialPhaseFalse/'
    weight1 = np.loadtxt(modelPath + 'layer1W.txt')
    bias1 = np.loadtxt(modelPath + 'layer1B.txt')
    weight2 = np.loadtxt(modelPath + 'layer2W.txt')
    bias2 = np.loadtxt(modelPath + 'layer2B.txt')

    blocks = 128
    block_size = np.int(K_max/blocks)

    w = np.append(w, np.mean(w))
    # print(w)
    wOrdered = np.zeros((len(w)))
    wOrder = np.flip(np.argsort(w, kind = 'stable'))
    wOrder = np.ascontiguousarray(wOrder, dtype=np.float64)
    for i in range(0, len(w)):
        wOrdered[i] = w[int(wOrder[i])]
    EquatorIndex = np.where(wOrder == N)
    
    aOrder = np.zeros(int((N+1)*(N)/2)).astype(np.float64)
    if pseudoRandomSequence:
        increasingIndices = -1* np.ones((N + 1, N + 1))
        increasingIndex = 0
        for i in range(0, N):
            for j in range(i + 1, N):
                increasingIndices[i, j] = increasingIndex
                increasingIndex += 1

    aLowerNew = np.zeros((N + 1, N + 1))
    aUpperNew = np.zeros((N + 1, N + 1))
    aLowerNew[0:N, 0:N] = aLowerBountIn.copy()
    aUpperNew[0:N, 0:N] = aUppwerBountIn.copy()

    aUpperOrdered = np.zeros((N + 1, N + 1))
    aLowerOrdered = np.zeros((N + 1, N + 1))
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            if i != j:
                aUpperOrdered[i, j] = aUpperNew[int(wOrder[i]), int(wOrder[j])]
                aLowerOrdered[i, j] = aLowerNew[int(wOrder[i]), int(wOrder[j])]
                
    if pseudoRandomSequence:
        index = 0
        for i in range(0, N + 1):
            for j in range(i + 1, N + 1):
                aOrder[index] = increasingIndices[min(int(wOrder[i]), int(wOrder[j])), max(int(wOrder[i]), int(wOrder[j]))]
                index += 1

    vec_a_lower = np.zeros(int((N+1)*(N)/2)).astype(np.float64)
    vec_a_upper = np.zeros(int((N+1)*(N)/2)).astype(np.float64)
    vec_a_lower = aLowerOrdered[np.triu_indices(N + 1, k = 1)]
    vec_a_upper = aUpperOrdered[np.triu_indices(N + 1, k = 1)]

    l1wVec = weight1.flatten().astype(np.float64)
    l1bVec = bias1.flatten().astype(np.float64)
    l2wVec = weight2.flatten().astype(np.float64)
    l2bVec = bias2.flatten().astype(np.float64)

    a_save = np.zeros(K_max).astype(np.float64)

    if (int(seed) == 0):
        rand_data = np.random.random(int((N-1)*N/2.0*K_max)).astype(np.float64)
    else:
        rand_data = np.random.RandomState(int(seed)).uniform(size = int((N-1)*N/2.0*K_max))

    taskNN(drv.In(l1wVec), drv.In(l1bVec), drv.In(l2wVec), drv.In(l2bVec), np.intc(EquatorIndex), drv.In(wOrder), drv.In(rand_data), drv.Out(a_save), drv.In(wOrdered), 
        np.float64(h), np.intc(N), np.intc(M), drv.In(vec_a_lower), drv.In(vec_a_upper), np.intc(pseudoRandomSequence), drv.In(aOrder), grid=(blocks,1), block=(block_size,1,1))
    
    if min(a_save) == -1:
        print("Non sync case exists")
        return -1
    
    if K_max >= 1000:
        temp = np.sort(a_save)
        ll = int(K_max*0.005)
        uu = int(K_max*0.995)
        a_save = temp[ll-1:uu]
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save)/(K_max*0.99)
    else:
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save)/(K_max)

    return MOCU_val

def MOCU(K_max, w, N, h , M, T, aLowerBoundIn, aUpperBoundIn, seed):
    # seed = 0
    blocks = 128
    block_size = np.int(K_max/blocks)

    w = np.append(w, np.mean(w))

    a_save = np.zeros(K_max).astype(np.float64)

    vec_a_lower = np.zeros(N*N).astype(np.float64)
    vec_a_upper = np.zeros(N*N).astype(np.float64)

    vec_a_lower = np.reshape(aLowerBoundIn.copy(), N*N)
    vec_a_upper = np.reshape(aUpperBoundIn.copy(), N*N)

    a = np.zeros((N+1)*(N+1)).astype(np.float64)

    if (int(seed) == 0):
        rand_data = np.random.random(int((N-1)*N/2.0*K_max)).astype(np.float64)
    else:
        rand_data = np.random.RandomState(int(seed)).uniform(size = int((N-1)*N/2.0*K_max))

    task(drv.In(a), drv.In(rand_data), drv.Out(a_save), drv.In(w), 
        np.float64(h), np.intc(N), np.intc(M), drv.In(vec_a_lower), 
        drv.In(vec_a_upper), grid=(blocks,1), block=(block_size,1,1))

    # print("a_save")
    # print(a_save)

    if min(a_save) == -1:
        print("Non sync case exists")
        return -1
    
    if K_max >= 1000:
        temp = np.sort(a_save)
        ll = int(K_max*0.005)
        uu = int(K_max*0.995)
        a_save = temp[ll-1:uu]
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save)/(K_max*0.99)

    else:
        a_star = max(a_save)
        MOCU_val = sum(a_star - a_save)/(K_max)

    return MOCU_val