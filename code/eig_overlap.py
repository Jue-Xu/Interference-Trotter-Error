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

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import colorsys
# # from utils import *
# # color_cycle = ["#E64B35FF", "#47B167FF", "#0A75C7FF", "#F39B7FFF", "#70699eFF", "#4DBBD5FF", "#FFAA00FF"]
# color_cycle = ["#b5423dFF", "#405977FF", "#616c3aFF", "#e3a13aFF", "#7a2c29FF", "#253a6aFF", "#8b9951FF"]
# color_cycle = [color[:-2] + "FF" for color in color_cycle]
# # Function to lighten a color
# def lighten_color(color, amount=0.3):
#     # Convert color from hexadecimal to RGB
#     r, g, b, a = tuple(int(color[i:i+2], 16) for i in (1, 3, 5, 7))
#     # Convert RGB to HLS
#     h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
#     # Lighten the luminance component
#     l = min(1, l + amount)
#     # Convert HLS back to RGB
#     r, g, b = tuple(round(c * 255) for c in colorsys.hls_to_rgb(h, l, s))
#     # Convert RGB back to hexadecimal
#     new_color = f"#{r:02x}{g:02x}{b:02x}{a:02x}"
#     return new_color

# color_cycle_light = [lighten_color(color, 0.3) for color in color_cycle]
# # color_cycle_light = [color[:-2] + "60" for color in color_cycle]
# colors = mpl.cycler(mfc=color_cycle_light, color=color_cycle, markeredgecolor=color_cycle)

# mpl.rc('axes', prop_cycle=colors)
# plt.rcParams['font.family'] = 'Helvetica'
# # plt.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['axes.linewidth'] = 1.5
# mpl.rcParams["xtick.direction"] = "in"
# mpl.rcParams["ytick.direction"] = "in"
# mpl.rcParams['xtick.major.width'] = 1.5
# mpl.rcParams['ytick.major.width'] = 1.5
# mpl.rcParams['ytick.minor.width'] = 1.5
# mpl.rcParams['lines.markersize'] = 10
# mpl.rcParams['legend.frameon'] = True
# # plt.rcParams['lines.markeredgecolor'] = 'k'
# mpl.rcParams['lines.linewidth'] = 1.5
# mpl.rcParams['lines.markeredgewidth'] = 1.0
# mpl.rcParams['figure.dpi'] = 100

# SMALL_SIZE = 14
# MEDIUM_SIZE = 15  #default 10
# LARGE_SIZE = 20
# MARKER_SIZE = 10

# plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
# plt.rc('axes', titlesize=LARGE_SIZE+2)  # fontsize of the axes title
# plt.rc('axes', labelsize=LARGE_SIZE)  # fontsize of the x and y labels
# plt.rc('xtick', labelsize=LARGE_SIZE)  # fontsize of the tick labels
# plt.rc('ytick', labelsize=LARGE_SIZE)  # fontsize of the tick labels
# plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
# plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

from sklearn.linear_model import LinearRegression

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

def sum_commutator(H_list):
    R = 0
    k = len(H_list)
    for i in range(k):
        for j in range(i):
            R += commutator(H_list[j], H_list[i])
    return R

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

def Delta_interference(H, R, h, T):
    r = 100
    # T = 1e6
    # for k in range(r):
    #     sum += expm(-1.0j * H * dt * (r-k)) @ (h*R ) @ expm(-1.0j * (H + h * R) * dt * k)
    # bad numerical precision! Now how to implement it: when T is large the difference between e^-iHT and e^-iH
    Delta = expm(-1.0j * H * T) - expm(-1.0j * (H+h*R) * T)
    T0 = 0.1 / spectral_norm(Delta) * T
    Delta = expm(-1.0j * H * T0) - expm(-1.0j * (H+h*R) * T0)
    return Delta / T0

def interference_bound(H_list, n, T, r):
    H = sum(H_list)
    R = -1.0j * sum_commutator(H_list) / 2
    h = T/r
    return spectral_norm(Delta_interference(H, R, h, T)) * T +  sum_second_order_commutator_norm(H_list) * (T/r)**3 * r + spectral_norm(R) * h

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

def bipartite_search_2term(error, H_list, H0_list, n, T, left_r, right_r, eps):
    if right_r < 0:
        right_r = left_r + 1
        while (error(H_list, H0_list, n, T, right_r) > eps):
            right_r *= 8
    while not (error(H_list, H0_list, n, T, left_r) <= eps or right_r == left_r + 1):
        print(left_r, right_r)
        mid = (left_r + right_r) // 2
        if (error(H_list, H0_list, n, T, mid) >= eps):
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

def approx_interference_bound(H_list, H0_list, n_qubit, T, r):
    R = sum_commutator(H_list)
    R0 = sum_commutator(H0_list)
    R1 = R - R0
    return trotter_error(H0_list, n_qubit, T, r) + T**2/2 * spectral_norm(R1) / r

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

def eig_renormalized(H_list, T = []):
    n = len(H_list)
    if T ==[]:
        T = np.zeros_like(H_list[0])
        for i in range(n):
            for j in range(i):
                T += commutator(H_list[j], H_list[i])
    R = sum(H_list)
    eigenvalues, eigenvectors = eig(R)
    renorm_T = np.zeros_like(T)
    for i in range(len(eigenvalues)):
        for j in range(len(eigenvalues)):
            li = eigenvalues[i]
            lj = eigenvalues[j]
            vi = eigenvectors[i]
            vj = eigenvectors[j]
            tij = vi @ T @ vj.conj().transpose()
            if (abs(li - lj)>1e-10):
                renorm_T += tij / (li-lj) * vi.conj().transpose() @ vj
    return T, renorm_T

def log_linear_regression(dict_r):
    # Sample dataset
    X = np.array(list(dict_r.keys()))
    X = np.log(X)
    X = X.reshape(-1,1)
    y = np.array(list(dict_r.values()))
    y = np.log(y)

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)

    # Output the coefficients
    coefficients = model.coef_
    intercept = model.intercept_

    print("Coefficients:", coefficients)
    print("Intercept:", intercept)
    return coefficients, intercept

# n_max = 10
# r_max = 100
# decay_coef = 1
# T_max = 100
# eps = 1e-5
# T = 10
# r_list = []
# error_list = {}

# mode_find_r_range_n = False
# mode_find_error_range_r = False
# mode_find_error_range_T = False
# mode_find_error_range_n = True

# if mode_find_error_range_n:
#     r = 5000
#     for n in range(2, 10):
#         H_list = neighbor_heisenberg(n)
#         error_list[n] = trotter_error(H_list, n, T, r)
#     print(error_list)

# if mode_find_error_range_r:
#     for r in range(1, r_max):
#         n = 5
#         H_list = power_law_heisenberg(n, decay_coef)
#         error_list[r*100] = trotter_error(H_list, n, T, r*100)
#     print(error_list)
    
# if mode_find_error_range_T:
#     for T in range(1, T_max):
#         n = 5
#         r = 500
#         H_list = power_law_heisenberg(n, decay_coef)
#         error_list[T] = trotter_error(H_list, n, T, r)
#         print(error_list)

# if mode_find_r_range_n:
#     for n in range(3, n_max):
#         r = 1
#         k = 1
#         H_list = power_law_heisenberg(n, decay_coef)
#         while (True):
#             if (trotter_error(H_list, n, T, r+1) < eps):
#                 break
#             if (trotter_error(H_list, n, T, r+k) < eps):
#                 k = k // 2
#                 continue
#             r = r+k
#             if r == k * 2:
#                 k *= 2
#             print(r)
#         r_list.append(r)
#     print(r_list)