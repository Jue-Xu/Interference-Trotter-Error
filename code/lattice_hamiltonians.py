import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from numpy.linalg import norm

from matplotlib.colors import ListedColormap
# colors = mpl.cycler(color=["c", "m", "y", "r", "g", "b", "k"]) 
# color_cycle = ["#4DBBD5FF", "#E64B35FF", "#00A087FF", "#70699e", "#F39B7FFF", "#3C5488FF", "#7E6148FF","#DC0000FF",  "#91D1C2FF", "#B09C85FF", "#923a3a", "#8491B4FF"]
# color_cycle = ["#E9002D", "#00B000", "#009ADE", "#FFAA00", "#6E005F"]
color_cycle = ["#E64B35FF", "#47B167", "#0A75C7", "#F39B7FFF", "#70699e"]
colors = mpl.cycler(color=color_cycle, alpha=[.9] * len(color_cycle)) 
# import scienceplots
# plt.style.use(['science', 'nature'])

mpl.rc('axes', prop_cycle=colors)
# mpl.use('ps')
# mpl.use("pgf")
# mpl.rc("pgf", texsystem="pdflatex", preamble=r"\usepackage{color}")

# mpl.rc('text',usetex=True)
# mpl.rc('text.latex', preamble='\\usepackage{color}')
# mpl.rc('axes', grid=True, edgecolor='k', prop_cycle=colors)
plt.rcParams['font.family'] = 'Helvetica'
# plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgecolor'] = 'k'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['figure.dpi'] = 150

SMALL_SIZE = 14
MEDIUM_SIZE = 18  #default 10
LARGE_SIZE = 24

plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE+2)  # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=LARGE_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=LARGE_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

import numpy as np
from cmath import exp
from scipy import sparse
from scipy.linalg import expm
from numpy.linalg import matrix_power
import scipy.sparse.linalg as ssla

import multiprocessing
import openfermion as of
# from qiskit.opflow import I, PauliOp, PauliSumOp, X, Y, Z
from qiskit.quantum_info.operators import Operator, Pauli
import random

def sum_pauli_ops(pauli_strs, coeff):
    """
    Sum a list of Pauli operators.

    Args:
        pauli_ops (list): A list of Pauli operators.

    Returns:
        sum_op (Operator): The sum of the Pauli operators.

    """
    return sum([coeff[index] * Operator(Pauli(item)) for index, item in enumerate(pauli_strs)])  

def rotate_str(string):
    """
    Shift a string by a given amount.

    Args:
        string (str): The string to be shifted.
        shift (int): The amount by which the string is shifted.

    Returns:
        shifted_string (str): The shifted string.

    """
    return [string[-shift:] + string[:-shift] for shift in range(len(string)) ]

def n_choose_2(n):
    """_summary_

    Args:
        n (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if n < 2:
        raise ValueError("n must be greater than or equal to 2.")

    d = {}
    for i in range(n):
        # print(i)
        for j in range(i+1, n):
            # print(f'j={j},n={n}')
            b_list = ['0'] * n
            b_list[i] = '1'
            b_list[j] = '1'
            b_str = ''.join(b_list)
            d[b_str] = abs(j-i)

    # print('d:', d)
    return d

class transverse_field_ising_1d:
    def __init__(self, n, J, h, verbose=False):
        """
        Constructs the Hamiltonian for the 1D transverse-field Ising model using Qiskit.

        Args:
            n (int): Number of spins in the chain.
            J (float): Coupling constant determining the interaction strength between neighboring spins.
            h (float): Strength of the transverse magnetic field.

        Returns:
            H (Operator): The Hamiltonian operator.

        """
        # shift the string by one to the right
        ZZ = 'ZZ' + (n-2) * 'I'
        XI = 'X' + (n-1) * 'I'
        self.interaction = rotate_str(ZZ)
        self.transverse = rotate_str(XI)

        if verbose: 
            print('---------Transverse-field Ising Hamiltonian---------')
            print(f'n={n}, J={J}, h={h}')
            print('interaction:', self.interaction)
            print('transverse', self.transverse)

        self.J = J; self.h = h
        self.zz = sum_pauli_ops(self.interaction, [self.J]*len(self.interaction)).to_matrix()
        self.x = sum_pauli_ops(self.transverse, [self.h]*len(self.transverse)).to_matrix()
        self.H = self.zz + self.x
        self.H_matrix= self.H

class heisenberg_1d:
    def __init__(self, n, Jx, Jy, Jz, h, verbose=False):
        '''
        Constructs the Hamiltonian for the 1D Heisenberg model using Qiskit.
        Args:
            n (int): Number of spins in the chain.
            Jx (float): Coupling constant determining the interaction strength between neighboring spins.
            Jy (float): Coupling constant determining the interaction strength between neighboring spins.     
            Jz (float): Coupling constant determining the interaction strength between neighboring spins.
            h: Strength of the transverse magnetic field.

        Returns:
            H (Operator): The Hamiltonian operator.
        '''
        self.n = n; self.Jx = Jx; self.Jy = Jy; self.Jz = Jz; self.h = h
        XX = 'XX' + (n-2) * 'I'
        YY = 'YY' + (n-2) * 'I'
        ZZ = 'ZZ' + (n-2) * 'I'
        ZI = 'Z' + (n-1) * 'I'
        XI = 'X' + (n-1) * 'I'
        self.inter_xx = rotate_str(XX)
        self.inter_yy = rotate_str(YY)
        self.inter_zz = rotate_str(ZZ)
        self.external =  rotate_str(ZI)

        if verbose: 
            print('---------Heisenberg Hamiltonian---------')
            print('inter_xx:', self.inter_xx)
            print('inter_yy:', self.inter_yy)
            print('inter_zz:', self.inter_zz)
            print('external', self.external)

        self.xx = sum_pauli_ops(self.inter_xx, [self.Jx]*len(self.inter_xx))
        self.yy = sum_pauli_ops(self.inter_yy, [self.Jy]*len(self.inter_yy))
        self.zz = sum_pauli_ops(self.inter_zz, [self.Jz]*len(self.inter_zz))
        self.z  = sum_pauli_ops(self.external, [self.h]*len(self.external))

        self.H = self.xx + self.yy + self.zz + self.z

        # return H

    def random_heisenberg(self, verbose=False):
        if verbose: print('random heisenberg')
        self.H_random = sum_pauli_ops(self.inter_xx, [random.random()]*len(self.inter_xx)).to_matrix() + sum_pauli_ops(self.inter_yy, [random.random()]*len(self.inter_yy)).to_matrix() + sum_pauli_ops(self.inter_zz, [random.random()]*len(self.inter_zz)).to_matrix() + sum_pauli_ops(self.external, [random.random()]*len(self.external)).to_matrix() 

    def power_law(self, alpha, verbose=False):
        # Heisenberg Hamiltonian with power law decay interaction
        # generate all bitstrings of length n with 2 ones

        self.alpha = alpha
        self.all_pow_inter = n_choose_2(self.n)
        if verbose: 
            print(f'---------Heisenberg Hamiltonian with Power law decaying interaction (alpha: {alpha})---------')
            print('all_pow_inter: ', self.all_pow_inter)

        self.pow_xx_str = [i.replace('1', 'X').replace('0', 'I') for i in self.all_pow_inter]
        self.pow_yy_str = [i.replace('1', 'Y').replace('0', 'I') for i in self.all_pow_inter]
        self.pow_zz_str = [i.replace('1', 'Z').replace('0', 'I') for i in self.all_pow_inter]

        self.pow_xx = sum_pauli_ops(self.pow_xx_str, [1/self.all_pow_inter[i]**self.alpha for i in self.all_pow_inter]).to_matrix()
        self.pow_yy = sum_pauli_ops(self.pow_yy_str, [1/self.all_pow_inter[i]**self.alpha for i in self.all_pow_inter]).to_matrix()
        self.pow_zz = sum_pauli_ops(self.pow_zz_str, [1/self.all_pow_inter[i]**self.alpha for i in self.all_pow_inter]).to_matrix()
        self.H_pow = self.pow_xx + self.pow_yy + self.pow_zz + self.z

    def partition(self, method, verbose=False):
        """
        Partitions the Hamiltonian into two parts, the interaction and the external field part.

        Args:
            method (int): 0 for interaction part and 1 for external field part.
        Returns:
            H (Operator): The Hamiltonian operator.

        """
        if method == 'parity':
            self.inter_xx_even = self.inter_xx[::2]
            self.inter_xx_odd = self.inter_xx[1::2]
            self.inter_yy_even = self.inter_yy[::2]
            self.inter_yy_odd = self.inter_yy[1::2]
            self.inter_zz_even = self.inter_zz[::2]
            self.inter_zz_odd = self.inter_zz[1::2]
            self.external_even = self.external[::2]
            self.external_odd = self.external[1::2]

            self.even = sum_pauli_ops(self.inter_xx_even, [self.Jx]*len(self.inter_xx_even)) + sum_pauli_ops(self.inter_yy_even, [self.Jy]*len(self.inter_xx_even)) + sum_pauli_ops(self.inter_zz_even, [self.Jz]*len(self.inter_zz_even)) + sum_pauli_ops(self.external_even, [self.h]*len(self.external_even)) 

            self.odd = sum_pauli_ops(self.inter_xx_odd, [self.Jx]*len(self.inter_xx_odd)) + sum_pauli_ops(self.inter_yy_odd, [self.Jy]*len(self.inter_yy_odd)) + sum_pauli_ops(self.inter_zz_odd, [self.Jz]*len(self.inter_zz_odd)) + sum_pauli_ops(self.external_odd, [self.h]*len(self.external_odd)) 

            if verbose:
                print('---------Partitioned Hamiltonian---------')
                print('inter_xx_even:', self.inter_xx_even)
                print('inter_xx_odd:', self.inter_xx_odd)
                print('external_even', self.external_even)
                print('external_odd', self.external_odd)
        elif method == 'tri':
            self.inter_xx_0 = self.inter_xx[::3]
            self.inter_xx_1 = self.inter_xx[1::3]
            self.inter_xx_2 = self.inter_xx[2::3]
            self.inter_yy_0 = self.inter_yy[::3]
            self.inter_yy_1 = self.inter_yy[1::3]
            self.inter_yy_2 = self.inter_yy[2::3]
            self.inter_zz_0 = self.inter_zz[::3]
            self.inter_zz_1 = self.inter_zz[1::3]
            self.inter_zz_2 = self.inter_zz[2::3]
            self.external_0 = self.external[::3]
            self.external_1 = self.external[1::3]
            self.external_2 = self.external[2::3]

            self.term0 = sum_pauli_ops(self.inter_xx_0, [self.Jx]*len(self.inter_xx_0)) + sum_pauli_ops(self.inter_yy_0, [self.Jy]*len(self.inter_yy_0)) + sum_pauli_ops(self.inter_zz_0, [self.Jz]*len(self.inter_zz_0)) + sum_pauli_ops(self.external_0, [self.h]*len(self.external_0)) 

            self.term1 = sum_pauli_ops(self.inter_xx_1, [self.Jx]*len(self.inter_xx_1)) + sum_pauli_ops(self.inter_yy_1, [self.Jy]*len(self.inter_yy_1)) + sum_pauli_ops(self.inter_zz_1, [self.Jz]*len(self.inter_zz_1)) + sum_pauli_ops(self.external_1, [self.h]*len(self.external_1)) 

            self.term2 = sum_pauli_ops(self.inter_xx_2, [self.Jx]*len(self.inter_xx_2)) + sum_pauli_ops(self.inter_yy_2, [self.Jy]*len(self.inter_yy_2)) + sum_pauli_ops(self.inter_zz_2, [self.Jz]*len(self.inter_zz_2)) + sum_pauli_ops(self.external_2, [self.h]*len(self.external_2)) 


        elif method == 'xyz':
            print('xyz')
        else:
            raise ValueError('method is not defined')
        # return 


"""Define the Hubbard Hamiltonian by OpenFermion."""
class hubbard_openfermion:
    def __init__(self, nsites, U, J=-1.0, pbc=False, verbose=False):
        # Each site has two spins.
        self.n_qubits = 2 * nsites

        def fop_2_sparse(fops):
            return [of.get_sparse_operator(fop, n_qubits=self.n_qubits).todense() for fop in fops ]

        # One-body (hopping) terms.
        self.one_body_fops = [op + of.hermitian_conjugated(op) for op in (
                of.FermionOperator(((i, 1), (i + 2, 0)), coefficient=J) for i in range(self.n_qubits - 2))]
        self.one_body_L = len(self.one_body_fops)
        self.one_body_sparse = fop_2_sparse(self.one_body_fops)

        # Two-body (charge-charge) terms.
        self.two_body_fops = [
            of.FermionOperator(((i, 1), (i, 0), (i + 1, 1), (i + 1, 0)), coefficient=U)
            for i in range(0, self.n_qubits, 2)]
        self.two_body_sparse = fop_2_sparse(self.two_body_fops)

        self.h_fop = of.fermi_hubbard(1, nsites, tunneling=-J, coulomb=U, periodic=pbc)
        self.h_sparse = of.get_sparse_operator(self.h_fop)
        self.ground_energy, self.ground_state = of.get_ground_state(self.h_sparse)
        assert sum(self.one_body_fops) + sum(self.two_body_fops) == self.h_fop
        if verbose: 
            print('one_body_terms: \n', self.one_body_fops)
            print('one_body_L: ', self.one_body_L)
            # print('one_body[0]: \n', self.one_body_sparse[0])
            print('one_body[0]: \n', of.get_sparse_operator(self.one_body_fops[0]))
            # print('sparse two-body: \n', of.get_sparse_operator(sum(self.two_body_fops)))
            # print('sparse two-body[0]: \n', self.two_body_sparse[0])
            # print('ground energy: \n', self.ground_energy)
        # return ground_energy

        self.one_body_01 = [term for index, term in enumerate(self.one_body_fops) if index % 4 == 0 or index % 4 == 1]
        self.one_body_01_sparse = fop_2_sparse(self.one_body_01)
        # print(self.one_body_01)
        self.one_body_23 = [term for index, term in enumerate(self.one_body_fops) if index % 4 == 2 or index % 4 == 3]
        self.one_body_23_sparse = fop_2_sparse(self.one_body_23)
        # print(self.one_body_23)
        assert sum(self.one_body_01) + sum(self.one_body_23) == sum(self.one_body_fops)

        self.one_body_0 = [term for index, term in enumerate(self.one_body_fops) if index % 3 == 0]
        self.one_body_1 = [term for index, term in enumerate(self.one_body_fops) if index % 3 == 1]
        self.one_body_2 = [term for index, term in enumerate(self.one_body_fops) if index % 3 == 2]

        assert sum(self.one_body_0) + sum(self.one_body_1)  + sum(self.one_body_2) == sum(self.one_body_fops)

        self.one_body_0_sparse = [term for index, term in enumerate(self.one_body_sparse) if index % 3 == 0]
        self.one_body_1_sparse = [term for index, term in enumerate(self.one_body_sparse) if index % 3 == 1]
        self.one_body_2_sparse = [term for index, term in enumerate(self.one_body_sparse) if index % 3 == 2]

def commutator(A, B):
    return A @ B - B @ A

def op_error(exact, approx, norm='spectral'):
    ''' 
    Frobenius norm of the difference between the exact and approximated operator
    input:
        exact: exact operator
        approx: approximated operator
    return: error of the operator
    '''
    if norm == 'fro':
        return np.linalg.norm(exact - approx)
    elif norm == 'spectral':
        return np.linalg.norm(exact - approx, ord=2)
    elif norm == 'HS':
        return abs(np.trace(exact.H @ approx))
    else:
        raise ValueError('norm is not defined')
    # return np.linalg.norm(exact - approx)/len(exact)

# matrix product of a list of matrices
def unitary_matrix_product(list_herm_matrices, t=1):
    ''' 
    matrix product of a list of unitary matrices exp(itH)
    input: 
        list_herm_matrices: a list of Hermitian matrices
        t: time
    return: the product of the corresponding matrices
    '''
    # product = expm(-1j * t * list_herm_matrices[0])
    # for i in range(1, len(list_herm_matrices)):
    #     product = product @ expm(-1j * t * list_herm_matrices[i])

    product = ssla.expm(-1j * t * list_herm_matrices[0])
    for i in range(1, len(list_herm_matrices)):
        product = product.dot(ssla.expm(-1j * t * list_herm_matrices[i]))

    return product

def data_plot(x, y, marker, label, alpha=1, linewidth=1, loglog=True):
    if loglog:
        plt.loglog(x, y, marker, label=label, linewidth=linewidth, markeredgecolor='black', markeredgewidth=0.5, alpha=alpha)
    else:
        plt.plot(x, y, marker, label=label, linewidth=linewidth, markeredgecolor='black', markeredgewidth=0.5, alpha=alpha)


# def data_plot(ax, x, y, marker, label, alpha=1, linewidth=1, loglog=True):
#     if loglog:
#         ax.loglog(x, y, marker, label=label, linewidth=linewidth, markeredgecolor='black', markeredgewidth=0.5, alpha=alpha)
#     else:
#         ax.plot(x, y, marker, label=label, linewidth=linewidth, markeredgecolor='black', markeredgewidth=0.5, alpha=alpha)

def matrix_plot(M):
    fig, ax = plt.subplots()
    real_matrix = np.real(M)
    # Plot the real part using a colormap
    ax.imshow(real_matrix, cmap='RdYlBu', interpolation='nearest', origin='upper')
    # Create grid lines
    ax.grid(True, which='both', color='black', linewidth=1)
    # Add color bar for reference
    cbar = plt.colorbar(ax.imshow(real_matrix, cmap='RdYlBu', interpolation='nearest', origin='upper'), ax=ax, orientation='vertical')
    cbar.set_label('Real Part')
    # Add labels to the x and y axes
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # Show the plot
    plt.title('Complex Matrix with Grid')
    plt.show()

def norm(A, ord='spectral'):
    if ord == 'fro':
        return np.linalg.norm(A)
    elif ord == 'spectral':
        return np.linalg.norm(A, ord=2)
    else:
        raise ValueError('norm is not defined')

def interference_bound(H, t, r):
    # Layden_2022_First-Order Trotter Error from a Second-Order Perspective
    try:
        assert len(H) == 2
    except:
        raise ValueError('The Hamiltonian contains not exactly 2 terms')

    h1 = H[0]
    h2 = H[1]
    C1 = min(norm(h1), norm(h2))
    C2 = 0.5 * norm(commutator(h1, h2))
    S = [norm(commutator(h1, commutator(h1, h2))), norm(commutator(h2, commutator(h2, h1)))]
    C3 = 1 / 12 * (min(S) + 0.5 * max(S))
    e1 = C1 * t / r
    e2 = C2 * t**2 / r
    e3 = C3 * t**3 / r**2
    bound = min(e2, e1 + e3, 2)
    # bound = min(e2, e1 + e3, 2 * len(h1))

    return bound, e1, e2, e3


def triangle_bound(h, k, t, r):
    L = len(h)
    if k == 1:
        if L == 2:
            raise ValueError('k=1 is not defined for L=2')
        elif L == 3:
            c = norm(commutator(h[0], h[1])) + norm(commutator(h[1], h[2])) + norm(commutator(h[2], h[0]))
            error = c * t**2 / (2*r) 
    return error

def tight_bound(h, order: int, t: float, r: int):
    L = len(h)
    if order == 1:
        a_comm = 0
        for i in range(0, L):
            temp = np.zeros(h[0].shape, dtype=complex)
            for j in range(i + 1, L):
                temp += commutator(h[i], h[j])
            a_comm += norm(temp)
        error = a_comm * t**2 / (2*r)
    elif order == 2:
        c1 = 0
        c2 = 0
        for i in range(0, L):
            temp = np.zeros(h[0].shape, dtype=complex)
            for j in range(i + 1, L):
                temp += h[j]
            # h_sum3 = sum(h[k] for k in range(i+1, L))
            # print(h_sum3.shape)
            # h_sum2 = sum(h[k] for k in range(i+1, L))
            c1 += norm(commutator(temp, commutator(temp, h[i]))) 
            # c1 = norm(commutator(h[0]+h[1], commutator(h[1]+h[2], h[0]))) + norm(commutator(h[2], commutator(h[2], h[1])))
            # c2 = norm(commutator(h[0], commutator(h[0],h[1]+h[2]))) + norm(commutator(h[1], commutator(h[1], h[2])))
            c2 += norm(commutator(h[i], commutator(h[i], temp)))
        error = c1 * t**3 / r**2 / 12 + c2 *  t**3 / r**2 / 24 
    else: 
        raise ValueError(f'higer order (order={order}) is not defined')

    return error

def analytic_bound(H, k, t, r):
    L = len(H)
    Lambda = max([norm(h) for h in H])

    return (2 * L * 5**(k-1) * Lambda * t)**(2*k+1)/(3*r**(2*k)) * exp((2*L*5**(k-1)*Lambda*t)/r)
