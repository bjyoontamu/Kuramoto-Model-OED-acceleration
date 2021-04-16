import numpy as np
import time

def determineSyncN(w, h, N, M, a):
    # if D=1, then sync
    # if D=0, then non-sync
    D = 0
    theta = np.zeros(N)
    theta_old = np.zeros(N)

    diff_t = np.zeros((M,N))

    for i in range(N):
        theta[i] = 0.0

    theta_old = theta.copy()

    F = np.zeros(N)
    t = 0.0

    for k in range(M):
        for i in range(N):
            F[i] = w[i] + np.sum(a[i,:]*np.sin(theta - theta[i]))
        k1 = h*F
        theta = theta_old + k1/2.0

        for i in range(N):
            F[i] = w[i] + np.sum(a[i,:]*np.sin(theta - theta[i]))
        k2 = h*F
        theta = theta_old + k2/2.0

        for i in range(N):
            F[i] = w[i] + np.sum(a[i,:]*np.sin(theta - theta[i]))
        k3 = h*F
        theta = theta_old + k3

        for i in range(N):
            F[i] = w[i] + np.sum(a[i,:]*np.sin(theta - theta[i]))
        k4 = h*F
        theta = theta_old + 1.0/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)


        diff_t[k,:] = (theta - theta_old)

        theta = np.remainder(theta,2.0*np.pi)
        theta_old = theta.copy()

        t = t+h

    tol = np.max(diff_t[int(M/2) - 1:,:]) - np.min(diff_t[int(M/2)-1:,:])

    if tol <= 10**(-3):
        D = 1

    return D