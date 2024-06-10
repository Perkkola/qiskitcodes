# Prepare input data
import random
import os
import math
import numpy as np
import pandas as pd
from functools import reduce
from collections import OrderedDict
from scipy import sparse
from scipy import optimize
from sklearn.preprocessing import StandardScaler, normalize
from scipy.sparse.linalg import expm, expm_multiply
from sklearn.decomposition import PCA

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, AncillaRegister
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit.quantum_info import Statevector
from qiskit.compiler import transpile
# from qiskit.primitives import Estimator
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.options import ExecutionOptions, Options
from qiskit.quantum_info import DensityMatrix,partial_trace, SparsePauliOp


#This is used for calculating the 2^-j values
#Very naive approach
def calcBinaryMatrix(n: int):
    matrixSize = int((2 ** (n / 2)) * (2 ** (n / 2)))
    y = [[]] * matrixSize
    for i in range(matrixSize):
        sum = 0
        for j in range(n):
            bit_value = 2 ** -(n - j)
            sum += bit_value * ((-1) ** (i >> j))
        y[i] = sum
    return y

#This is used to calculate the estimate values from the binary matrix e.g. the digitalization of the
#data
#Also very naice approach
def estimate(xx: float, n: int, matrix):
    half = int(len(matrix) / 2)
    difference = 2 ** 31
    index = 0
    if xx >= 0: 
        for i in range(half):
            if np.abs(matrix[i] - xx) < difference:
                difference = np.abs(matrix[i] - xx)
                index = i
    else:
        for i in range(half, len(matrix)):
            if np.abs(matrix[i] - xx) < difference:
                difference = np.abs(matrix[i] - xx)
                index = i
    estimated = [0] * n
    for i in range(n):
        estimated[i] = (index >> i) & 1
    return estimated

#This produces the delta_hat operator. See the paper for reference
def createDeltaHat(a: float, n: int):
    Z = sparse.csc_matrix(np.array([[1, 0], [0, -1]]))
    I = sparse.csc_matrix(np.array([[1, 0], [0, 1]]))

    j = 2 ** -1
    first = Z
    for i in range(1, n):
        # first = first.tensor(I)
        first = sparse.kron(first, I)

    delta_hat = (a * j) * first

    for i in range(1, n):
        delta = I
        for k in range(1, n):
            if i == k:
                # delta = delta.tensor(Z)
                delta = sparse.kron(delta, Z)
            else:
                # delta = delta.tensor(I)
                delta = sparse.kron(delta, I)
        delta_hat += (a * 2 ** -(j + 1)) * delta
    # print(delta_hat)
    return delta_hat
        
#This is the unitary for endocing the data in the binary encoded data approach. See paper for reference
def createU_k(k_index: int, x_k: float):
    Z = sparse.csc_matrix(np.array([[1, 0], [0, -1]]))
    I = sparse.identity(2, format='csc')
    
    k = np.array([0] * 2 ** QPU_len)
    k[k_index] = 1
    k.shape = (2 ** QPU_len, 1)
    k = sparse.csc_matrix(k)
    k = k.dot(k.transpose())

    U = expm(-1j * x_k * sparse.kron(Z, k))
    return U

#This is the unitary for regression coefficients.
def createU_m(phi_m: float, m_index: int):
    Z = sparse.csc_matrix(np.array([[1, 0], [0, -1]]))
    

    global N_L
    global N_M

    I = sparse.identity(2 ** N_L, format='csc')

    m_projector = np.array([0] * 2 ** N_M)
    m_projector[m_index] = 1
    m_projector.shape = (2 ** N_M, 1)
    m_projector = sparse.csc_matrix(m_projector)
    m_projector = m_projector.dot(m_projector.transpose())
    U = expm(1j * phi_m * sparse.kron(I, sparse.kron(m_projector, Z)))
    return U


#The measurement operator
def create_x_index_list():
    global N_M
    x_index_list = []

    for i in range(1, 2 ** N_M):
        binary_string = ("{:0{width}b}".format(i, width=N_M))[::-1]
        indices = [i for i, x in enumerate(binary_string) if x == '1']
        x_index_list.append(indices)
    return x_index_list


def cut_counts(counts, bit_indexes):
    bit_indexes.sort(reverse=True) 
    new_counts = {}

    for key in counts:
        # if(key[-1] == '1' and key[-2] == '0'):
        if(key[-1] == '0'):
            # print(key)
            new_key = ''
            for index in bit_indexes:
                new_key += key[-2 - index]
            if new_key in new_counts:
                new_counts[new_key] += counts[key]
            else:
                new_counts[new_key] = counts[key]
    # print(counts)
    # print(new_counts)
    return new_counts

def post_select(counts, x_index_list):

    x_counts = cut_counts(counts, x_index_list)
    expval = 0
    for key, value in zip(x_counts.keys(), x_counts.values()):
        if(key.count('1') % 2 == 0):
            expval += value
        else:
            expval -= value
    return expval
# df = pd.read_csv("./Diabetes_dataset.csv")

# X = np.array(df.iloc[:,:-1])
# y = np.array(df.iloc[:,-1])

X = np.array([[-0.32741112, -0.11288069,  0.49650164],
       [-0.94268847, -0.78149813, -0.49440176],
       [ 0.68523899,  0.61829019, -1.32935529],
       [-1.25647971, -0.14910498, -0.25044557],
       [ 1.66252391, -0.78480779,  1.79644309],
       [ 0.42989295,  0.45376306,  0.21658276],
       [-0.61965493, -0.39914738, -0.33494265],
       [-0.54552144,  1.85889336,  0.67628493]])

y = np.array([ -8.02307406, -23.10019118,  16.79149797, -30.78951577,
        40.73946101,  10.53434892, -15.18438779, -13.3677773 ])

l = len(X) #Rows
m = len(X[0]) + 1 #Columns (including label)

N_M = int(np.ceil(np.log2(m))) #Binary length for column items
N_L = int(np.ceil(np.log2(l))) #Binary length for row items

QPU_len = N_M + N_L

data = np.empty((l, m))

for i in range(l):
    # print(data[i])
    data[i] = np.append(y[i], X[i]) #Reverse the data order
    # print(data[i])
    # exit()


squareSum = 0
data = data.transpose()

# #Standardize column-wise and normalize globally
for i in range(m):
    data[i] = data[i] - np.mean(data[i])
    data[i] = data[i] / np.std(data[i])
    for j in range(l):
        squareSum += np.square(data[i][j])

data = data.transpose()
squareSum = np.sqrt(squareSum)
data = data / squareSum

dataPadded = np.zeros((2**N_L, 2**N_M))
for i in range(l):
    dataPadded[i] = np.append(data[i], [0] * (int(2**N_M - m)))

data = dataPadded.flatten()

data_copy = np.copy(data)
for j in range(2 ** QPU_len):
    reversed_index = int(("{:0{width}b}".format(j, width=QPU_len))[::-1], 2)
    data[reversed_index] = data_copy[j]

epsilon = 10**-15
shots = 2 ** 17

print(QPU_len)

# M_hat = np.real(createM_hat().to_matrix())
sparse_U_k = [createU_k(i, data[i]) for i in range(len(data))]
sparse_U_k_product = reduce(lambda operator, product: operator.multiply(product), sparse_U_k)
# M_hat = Operator(createM_hat().toarray())
x_index_list = create_x_index_list()

#Function for NM optimizer
def run_circuit(phi):

    ar = AncillaRegister(1, 'ancilla')
    row_reg = QuantumRegister(N_L, 'l')
    col_reg = QuantumRegister(N_M, 'm')
    cr = ClassicalRegister(N_M + 1, 'cr')

    # psi = QuantumCircuit(ar, qpu, cr)
    psi = QuantumCircuit(ar, row_reg, col_reg,  cr)


    # estimated = np.copy(data)
    # squareSum = 0
    # # phi = phi[::-1]
    # for j in range(2 ** N_L):
    #     for i in range(2 ** N_M):
    #         index = j * 2 ** N_M + i
    #         reversed_index = int(("{:0{width}b}".format(index, width=QPU_len))[::-1], 2)
    #         estimated[reversed_index] = estimated[reversed_index]
    #         squareSum += np.square(estimated[reversed_index])

    # squareSum = np.sqrt(squareSum)
    # estimated = estimated / squareSum

    psi.initialize(data, [row_reg, col_reg])
    # ordering1 = [0, 3, 2, 1]
    # ordering2 = [0, 1, 2, 3]
    sparse_U_m = [createU_m(phi[i], i) for i in range(2 ** N_M)]
    # sparse_U_m_product = reduce(lambda operator, product: operator.multiply(product), sparse_U_m)

    # psi.h([x for x in range(QPU_len + 1)])

    # for U in sparse_U_k:
    #     psi.append(Operator(U.toarray()), [x for x in range(1 + QPU_len)])

    # # # psi.append(Operator(sparse_U_k_product.toarray()), [x for x in range(1 + QPU_len)])
    # psi.h(ar)
    # psi.measure(ar, cr[0])
    # psi.x(ar)
    psi.h(ar)
    for U in sparse_U_m:
        psi.append(Operator(U.toarray()), [x for x in range(1 + QPU_len)])
    # # psi.append(Operator(sparse_U_m_product.toarray()), [x for x in range(1 + QPU_len)])
    psi.h(ar)
    psi.measure(ar, cr[0])

    psi.h(col_reg)

    # hamiltonian = SparsePauliOp(['IIIII', 'IXIII', 'XIIII', 'XXIII'])
    for i in range(N_M):
        psi.measure(col_reg[i], cr[i + 1])
    # psi.measure_all()

    # print(psi)
    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    isa_qc = pm.run(psi)
    # isa_observables = hamiltonian.apply_layout(isa_qc.layout)

    # estimator = Estimator(backend=aer_sim, options={"default_precision": 0.005})
    # job = estimator.run([(isa_qc, isa_observables)])
    # pub_result = job.result()[0]
    # print(phi)
    # print(f"Expectation values: {pub_result.data.evs / math.pow(math.cos(phi[0]), 2)}")
    # exit()
    # return pub_result.data.evs / math.pow(math.cos(phi[0]), 2)
    # exit()
    # # print(isa_qc)
    with Session(backend=aer_sim) as session:
        sampler = Sampler(session=session)
        result = sampler.run([isa_qc], shots=shots).result()

    counts = result[0].data.cr.get_counts()
    return counts
    
def calc_expval(phi):
    # print(phi)
    expval = 1

    counts = run_circuit(phi)
    for x_list in x_index_list:
        expval += post_select(counts, x_list) / shots
    expval /= math.pow(math.cos(phi[0]), 2)
    # # expval *= 100
    print(phi, expval)
    exit()
    # expval = run_circuit(phi, M_hats[0])
    return expval

#The rest is basically for running the optimizer

init = [np.pi / 2]  * (2 ** N_M - 1)
init.insert(0, 3 * np.pi / 4, ) #Initial parameters
# init = [3.36172813, 1.57079631, 1.57079679, 0.22012573]
init = [np.pi, 0, np.pi / 2, np.pi / 2]
bounds = [(-np.pi, np.pi)] * (2 ** N_M - 1)
bounds.insert(0, ( np.pi / 2, 3 * np.pi / 2))
bounds = tuple(bounds) #Bounds

res = optimize.minimize(fun = calc_expval, x0 = init, method = 'Nelder-Mead', options={'maxiter' : 200, 'disp': True}, bounds = bounds)
print(res.x)
print(res.message)


    #     if("{:0{width}b}".format(i, width=QPU_len) not in states): statesWithIndex.append(("{:0{width}b}".format(i, width=QPU_len), 0))
    #     service = QiskitRuntimeService(channel='ibm_quantum',token='8b7dbd957b8397e509a9b18af70f77a2853e7f9ef6a7cec345300e9e530f654061389b4ddfb86d9668cffe36379349f2d6c6d3a470609f927b423b66fac16254')

