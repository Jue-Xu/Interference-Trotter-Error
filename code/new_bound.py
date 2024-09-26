from numpy.linalg import matrix_power, eig
import numpy as np
from copy import deepcopy

########## Boyang's code ##########
def commutator(A, B):
    return A@B - B@A

def sum_second_order_commutator_norm(H_list):
    sum = 0
    n = len(H_list)
    for gamma_1 in range(n):
        R = np.zeros_like(H_list[0])
        for gamma_3 in range(gamma_1+1, n):
            for gamma_2 in range(gamma_1+1, n):
                R += commutator(H_list[gamma_3], commutator(H_list[gamma_2], H_list[gamma_1])) / 12
        sum += spectral_norm(R)
    for gamma_1 in range(n):
        R = np.zeros_like(H_list[0])
        for gamma_2 in range(gamma_1+1, n):
            R += commutator(H_list[gamma_1], commutator(H_list[gamma_1], H_list[gamma_2])) / 24
        sum += spectral_norm(R)
    return sum

def spectral_norm(U0):
    return np.linalg.norm(U0, ord=2)

def near_diagonal(R, eigenvalues, eps):
    N = R.shape[0]
    # print(N)
    R_copy = deepcopy(R)
    for i in range(N):
        for j in range(N):
            if abs(eigenvalues[i] - eigenvalues[j]) > eps:
                R_copy[i][j] = 0
    return R_copy

def interference_bound_new(R, H, T, eps):
    N = R.shape[0]
    eigenvalues, eigenvectors = eig(H)
    R_norm = spectral_norm(R)
    delta_norm = spectral_norm(near_diagonal(R, eigenvalues, eps)) # eps refers to the spectral gap
    resid_norm = spectral_norm(R-near_diagonal(R, eigenvalues, eps))
    # print(R_norm, delta_norm, resid_norm)
    # return (resid_norm * min(1, (1 / (eps * T))) + delta_norm) 
    return (resid_norm *  (1 / (eps * T)) + delta_norm) 
########## Boyang's code ##########
