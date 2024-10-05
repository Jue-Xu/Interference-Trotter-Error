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


def interference_bound_new2(R, H, T):
    eigenvalues, eigenvectors = eig(H)
    near_norm = {}
    for i in range(-10, 5):        
        if i==-10:
            near_norm[i] = spectral_norm(near_diagonal(R, eigenvalues, 2 ** i))
        else:
            near_norm[i] = spectral_norm(near_diagonal(R, eigenvalues, 2 ** i)) - spectral_norm(near_diagonal(R, eigenvalues, 2 ** (i-1)))
    interference_R = 0
    for i in range(-10, 5):
        interference_R += near_norm[i] * min(1, 1/(T*(2**i)))
    return interference_R

def interference_bound_new3(R, H, dt):
    H_mat, R_mat = H, R
    HR_mat = H_mat + R_mat * dt / (2j) 
    HR_eigvals, HR_eigvecs = np.linalg.eigh(HR_mat)[0], np.linalg.eigh(HR_mat)[1]
    H_eigvals, H_eigvecs = np.linalg.eigh(H_mat)[0], np.linalg.eigh(H_mat)[1]

    eps = 1e-3
    dim = len(H_eigvals)
    DR =  np.zeros((dim, dim), dtype=complex)
    RR =  np.zeros((dim, dim), dtype=complex)

    B = H_eigvecs.T.conj() @ R_mat @ HR_eigvecs
    for j in range(dim):
        for k in range(dim):
            v, u = H_eigvecs[:, j], HR_eigvecs[:, k]
            # b_jk = v.T.conj() @ R_mat @ u
            # print('b_jk: ', b_jk)
            if abs(H_eigvals[j] - HR_eigvals[k]) < eps:
                DR += B[j,k] * np.outer(v, u.conj())
            else:
                RR += 1/(H_eigvals[j] - HR_eigvals[k]) * B[j,k] * np.outer(v, u.conj())

    print(f'||Delta(R)||={np.linalg.norm(DR, ord=2)}, ||R(R)||={np.linalg.norm(RR, ord=2)}')

    return np.linalg.norm(DR, ord=2), np.linalg.norm(RR, ord=2)