import numpy as np
from numpy import kron, sqrt, pi, arccos, cos, sin, exp
from numpy.linalg import norm
from numpy.linalg import matrix_power, eig
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import optimize
from matplotlib import pyplot as plt
import functools as ft
import random
import operator

def process_fidelity(U0, U1):
    return abs(np.trace(U0 @ U1.conj().transpose())) / len(U0)

def spectral_norm(U0):
    return np.linalg.norm(U0, ord=2)

def local_interaction(n, dict):
    R_list = [I] * n
    for i in dict:
        R_list[i] = dict[i]
    return ft.reduce(kron, R_list)

X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1, 0], [0, 1]])

H_tmp = []
def trotter_simulation(H_list, n, T, r):
    U1 = np.eye(2 ** n).astype(np.complex128)
    # print(H_list, T, r)
    for H in H_list:
        H_tmp.append(-1.0j * H * T / r)
        U1 = U1 @ expm(-1.0j * H * T / r)
    return matrix_power(U1, r)

def trotter_error(H_list, n_qubit, T, r):
    U_target = expm(-1.0j* sum(H_list) * T)
    U_impl = trotter_simulation(H_list, n_qubit, T, r)
    return spectral_norm(U_impl - U_target)

def commutator(A, B):
    return A@B - B@A

def bipartite_search(error, H_list, n, T, left_r, right_r, eps):
    if right_r < 0:
        right_r = left_r + 1
        while (error(H_list, n, T, right_r) > eps):
            right_r *= 8
    while not (error(H_list, n, T, left_r) <= eps or right_r == left_r + 1):
        print(left_r, right_r)
        mid = (left_r + right_r) // 2
        if (error(H_list, n, T, mid) >= eps):
            left_r = mid
        else:
            right_r = mid
    return left_r

def error(H_list, n, T, r):
    return 1.0/r

def neighbor_heisenberg(n_qubit):
    edges = []
    for i in range(n_qubit-1):
        edges.append((i, i+1))    
    XX = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    YY = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    ZZ = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    edges.append((0, n_qubit-1))

    for i, j in edges:
        XX += local_interaction(n_qubit, {i:X, j:X})
        YY += local_interaction(n_qubit, {i:Y, j:Y})
        ZZ += local_interaction(n_qubit, {i:Z, j:Z})
    return [XX, YY, ZZ]

def neighbor_lattice(n_qubit, interactions):
    edges = []
    for i in range(n_qubit-1):
        edges.append((i, i+1))    
    XX = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    YY = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    ZZ = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    edges.append((0, n_qubit-1))

    for i, j in edges:
        XX += local_interaction(n_qubit, {i:interactions[0][0], j:interactions[0][1]})
        YY += local_interaction(n_qubit, {i:interactions[1][0], j:interactions[1][1]})
        ZZ += local_interaction(n_qubit, {i:interactions[2][0], j:interactions[2][1]})
    return [XX, YY, ZZ]    

def neighbor_heisenberg_parity_grouping(n_qubit, interactions = [[X, X], [Y, Y], [Z, Z]]):
    edges = []
    for i in range(n_qubit-1):
        edges.append((i, i+1))
    edges.append((0, n_qubit-1))
    EVEN = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    ODD = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)

    for i, j in edges:
        ij_XYZ = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
        ij_XYZ += local_interaction(n_qubit, {i:interactions[0][0], j:interactions[0][1]})
        ij_XYZ += local_interaction(n_qubit, {i:interactions[1][0], j:interactions[1][1]})
        ij_XYZ += local_interaction(n_qubit, {i:interactions[2][0], j:interactions[2][1]})
        if i % 2 == 0:
            EVEN += ij_XYZ
        else:
            ODD += ij_XYZ

    # for i in range(n_qubit):
    #     ZZ += random.random() * local_interaction(n_qubit, {i:Z})
    return [EVEN, ODD]


def neighbor_heisenberg_tri_grouping(n_qubit):
    edges = []
    for i in range(n_qubit-1):
        edges.append((i, i+1))
    edges.append((0, n_qubit-1))
    EVEN = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    ODD = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    TRIAD = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128) 

    for i, j in edges:
        ij_XYZ = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
        ij_XYZ += local_interaction(n_qubit, {i:X, j:X})
        ij_XYZ += local_interaction(n_qubit, {i:Y, j:Y})
        ij_XYZ += local_interaction(n_qubit, {i:Z, j:Z})
        if i % 3 == 0:
            EVEN += ij_XYZ
        elif i%3 == 1:
            ODD += ij_XYZ
        else:
            TRIAD += ij_XYZ
    return [EVEN, ODD, TRIAD]

def power_law_heisenberg(n_qubit, r):
    XX = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128) 
    YY = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    ZZ = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    for i in range(n_qubit):
        for j in range(i):
            w = 1 / abs(j-i) ** r
            XX += local_interaction(n_qubit, {i:X, j:X}) * w
            YY += local_interaction(n_qubit, {i:Y, j:Y}) * w
            ZZ += local_interaction(n_qubit, {i:Z, j:Z}) * w
    for i in range(n_qubit):
        YY += 0 * local_interaction(n_qubit, {i:Y})
    return [XX, YY, ZZ]

def random_heisenberg(n_qubit):
    XX = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128) 
    YY = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    ZZ = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
    for i in range(n_qubit):
        for j in range(i):
            XX += local_interaction(n_qubit, {i:X, j:X}) * random.random()
            YY += local_interaction(n_qubit, {i:Y, j:Y}) * random.random()
            ZZ += local_interaction(n_qubit, {i:Z, j:Z}) * random.random()
    return [XX, YY, ZZ]    

def triangle_bound(H_list, n_qubit, T, r):
    sum_norm_commutator = 0
    k = len(H_list)
    for i in range(k):
        cumsum_H = np.zeros((2**n_qubit, 2**n_qubit)).astype(np.complex128)
        for j in range(i+1, k):
            cumsum_H += H_list[j]
        sum_norm_commutator += spectral_norm(commutator(H_list[i], cumsum_H))
    return 0.5 * T ** 2 * sum_norm_commutator / r

def accumulated_bound(H_list, n_qubit, T, r):
    step_error = trotter_error(H_list, n_qubit, T/r, 1)
    return step_error * r

def eig_overlap(H_list, T = []):
    n = len(H_list)
    if T ==[]:
        T = np.zeros_like(H_list[0])
        for i in range(n):
            for j in range(i):
                T += commutator(H_list[j], H_list[i])
    R = sum(H_list)
    eigenvalues, eigenvectors = eig(R)
    correlation = []
    for i in range(len(eigenvalues)):
        v = eigenvectors[:,i]
        v = v.conj().transpose()
        print(np.around(eigenvalues[i], decimals=6), np.around(v @ T @ v.conj().transpose(), decimals=6))
        correlation.append(abs(v @ T @ v.conj().transpose()))
    return max(correlation)/norm(T)

def eig_overlap_avg(H_list, T = []):
    n = len(H_list)
    if T ==[]:
        T = np.zeros_like(H_list[0])
        for i in range(n):
            for j in range(i):
                T += commutator(H_list[j], H_list[i])
    R = sum(H_list)
    eigenvalues, eigenvectors = eig(R)
    correlation = []
    for i in range(len(eigenvalues)):
        v = eigenvectors[:,i]
        v = v.conj().transpose()
        print(np.around(eigenvalues[i], decimals=6), np.around(v @ T @ v.conj().transpose(), decimals=6))
        correlation.append(abs(v @ T @ v.conj().transpose()))
    return np.average(correlation)

n_max = 10
r_max = 100
decay_coef = 1
T_max = 100
eps = 1e-5
T = 10
r_list = []
error_list = {}

mode_find_r_range_n = False
mode_find_error_range_r = False
mode_find_error_range_T = False
mode_find_error_range_n = True

if mode_find_error_range_n:
    r = 5000
    for n in range(2, 10):
        H_list = neighbor_heisenberg(n)
        error_list[n] = trotter_error(H_list, n, T, r)
    print(error_list)

if mode_find_error_range_r:
    for r in range(1, r_max):
        n = 5
        H_list = power_law_heisenberg(n, decay_coef)
        error_list[r*100] = trotter_error(H_list, n, T, r*100)
    print(error_list)
    
if mode_find_error_range_T:
    for T in range(1, T_max):
        n = 5
        r = 500
        H_list = power_law_heisenberg(n, decay_coef)
        error_list[T] = trotter_error(H_list, n, T, r)
        print(error_list)

if mode_find_r_range_n:
    for n in range(3, n_max):
        r = 1
        k = 1
        H_list = power_law_heisenberg(n, decay_coef)
        while (True):
            if (trotter_error(H_list, n, T, r+1) < eps):
                break
            if (trotter_error(H_list, n, T, r+k) < eps):
                k = k // 2
                continue
            r = r+k
            if r == k * 2:
                k *= 2
            print(r)
        r_list.append(r)
    print(r_list)