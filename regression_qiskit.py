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
from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, AncillaRegister
from qiskit.circuit.library import RZGate, MCMT
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit.transpiler import PassManager, CouplingMap, TransformationPass
from qiskit.quantum_info import Statevector
from qiskit.compiler import transpile
# from qiskit.primitives import Estimator
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.options import ExecutionOptions, Options
from qiskit.quantum_info import DensityMatrix,partial_trace, SparsePauliOp
from gray_synth import synth_cnot_phase_aam
import sys
np.set_printoptions(threshold=sys.maxsize)

def generate_cnots(arr):
    temp = []
    arr = np.array(arr).transpose().tolist()
    for i in range(2):
        for a in arr:
            cp = np.copy(a).tolist()
            cp.append(i)
            temp.append(cp)

    return np.array(temp).transpose().tolist()

def generate_thetas(arr):
    arr = np.array(arr) / 2
    arr = np.append(arr, arr * -1)
    return arr.tolist()

def flip_signs(cnots, thetas, x_index_list):
    arr = np.copy(cnots).transpose().tolist()
    for index, bit_list in enumerate(arr):
        xor_list = [bit_list[x] for x in x_index_list]
        xor_result = reduce(lambda a, b: a ^ b, xor_list)
        if xor_result == 1: thetas[index] *= -1
    return thetas


def create_phase_x_index_list(qubit_length):
    x_index_list = []

    for i in range(1, 2 ** qubit_length):
        binary_string = ("{:0{width}b}".format(i, width=qubit_length))[::-1]
        indices = [i + 1 for i, x in enumerate(binary_string) if x == '1']
        x_index_list.append(indices)
    return x_index_list

#This is the unitary for endocing the data in the binary encoded data approach. See paper for reference
def createU_k(circuit, data_arr):
    cnots = [[1, 1],
             [0, 1]]
    all_thetas = []
    for _ in range(QPU_len - 1):
        cnots = generate_cnots(cnots)
    
    for theta in data_arr:
        thetas = [theta/2, -theta/2]
        for _ in range(QPU_len - 1):
            thetas = generate_thetas(thetas)
        all_thetas.append(thetas)


    thetas = np.copy(all_thetas[-1])
    x_phase_index_lists = create_phase_x_index_list(QPU_len)[::-1]

    for thetas_cp, x_index_list in zip(all_thetas[:-1], x_phase_index_lists):
        flipped_thetas = flip_signs(cnots, np.copy(thetas_cp), x_index_list)
        thetas = np.array(thetas) + np.array(flipped_thetas)
    thetas = thetas.tolist()

    gray_qc = synth_cnot_phase_aam(cnots, thetas)
    circuit.append(gray_qc.to_gate(), [x for x in range(QPU_len + 2) if x != 1])
    circuit.barrier([x for x in range(QPU_len + 2)])

    # for i in range(len(data_arr)):
    #     bit_string = ("{:0{width}b}".format(i, width=QPU_len))[::-1]
    #     x_index_list = [i + 2 for i, x in enumerate(bit_string) if x == '0']
    #     mcrz_index_list = [x + 2 for x in range(QPU_len)]
    #     mcrz_index_list.append(0)
    #     for j in range(2):
    #         if len(x_index_list) > 0: circuit.x(x_index_list)
    #         if j == 0:
    #             rz = RZGate(data[i] * 2)
    #             mcrz = MCMT(rz, QPU_len, 1)
    #             circuit.append(mcrz, mcrz_index_list)
    #             # circuit.mcp(data_arr[i], [x + 1 for x in range(QPU_len)], 0)
    #             # # circuit.mcx([x + 1 for x in range(QPU_len)], 0)
    #             # circuit.x(0)
    #             # circuit.mcp(-data_arr[i], [x + 1 for x in range(QPU_len)], 0)
    #             # circuit.x(0)
    #             # # circuit.mcx([x + 1 for x in range(QPU_len)], 0)

    #     circuit.barrier([x for x in range(QPU_len + 2)])


        
#This is the unitary for regression coefficients.
def createU_m(circuit, col_reg, phi_array):
    global N_M
    # for i in range(len(phi_array)):
    #     bit_string = ("{:0{width}b}".format(i, width=N_M))[::-1]
    #     x_index_list = [i + 2 for i, x in enumerate(bit_string) if x == '0']
    #     mcrz_index_list = [x + 2 for x in range(N_M)]
    #     mcrz_index_list.append(1)
    #     for j in range(2):
    #         if len(x_index_list) > 0: circuit.x(x_index_list)
    #         if j == 0:
    #             rz = RZGate(-phi_array[i] * 2)
    #             mcrz = MCMT(rz, N_M, 1)
    #             circuit.append(mcrz, mcrz_index_list)
    #             # circuit.mcp(-phi_array[i], col_reg, 0)
    #             # # circuit.mcx(col_reg, 0)
    #             # circuit.x(0)
    #             # circuit.mcp(phi_array[i], col_reg, 0)
    #             # # circuit.mcx(col_reg, 0)
    #             # circuit.x(0)
    cnots = [[1, 1],
             [0, 1]]
    all_thetas = []
    for _ in range(N_M - 1):
        cnots = generate_cnots(cnots)
    
    for theta in phi_array:
        thetas = [-theta/2, theta/2]
        for _ in range(N_M - 1):
            thetas = generate_thetas(thetas)
        all_thetas.append(thetas)


    thetas = np.copy(all_thetas[-1])
    x_phase_index_lists = create_phase_x_index_list(N_M)[::-1]
    
    for thetas_cp, x_index_list in zip(all_thetas[:-1], x_phase_index_lists):
        flipped_thetas = flip_signs(cnots, np.copy(thetas_cp), x_index_list)
        thetas = np.array(thetas) + np.array(flipped_thetas)
    thetas = thetas.tolist()

    gray_qc = synth_cnot_phase_aam(cnots, thetas)
    circuit.append(gray_qc.to_gate(), [x for x in range(1, N_M + 2)])
    circuit.barrier([x for x in range(QPU_len + 2)])



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
    discarded = 0
    for key in counts:
        if(key[-1] == '1' and key[-2] == '0'):
            new_key = ''
            for index in bit_indexes:
                new_key += key[-3 - index]
            if new_key in new_counts:
                new_counts[new_key] += counts[key]
            else:
                new_counts[new_key] = counts[key]
        else:
            discarded += counts[key]
    new_shots = shots - discarded
    return new_counts, new_shots

def post_select(counts, x_index_list):

    x_counts, new_shots = cut_counts(counts, x_index_list)
    expval = 0
    for key, value in zip(x_counts.keys(), x_counts.values()):
        if(key.count('1') % 2 == 0):
            expval += value / new_shots
        else:
            expval -= value / new_shots
    return expval

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
# df = pd.read_csv("./Admission_Predict.csv")

# X = np.array(df.iloc[:,1:-1])
# y = np.array(df.iloc[:,-1])

# X, X_test, y, y_test = train_test_split(X, y, test_size=0.36, random_state=42)


l = len(X) #Rows
m = len(X[0]) + 1 #Columns (including label)

N_M = int(np.ceil(np.log2(m))) #Binary length for column items
N_L = int(np.ceil(np.log2(l))) #Binary length for row items

QPU_len = N_M + N_L

data = np.empty((l, m))

for i in range(l):
    data[i] = np.flip(np.append(X[i], y[i]))
    # data[i] = np.append(y[i], X[i])

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
shots = 2 ** 14
print(QPU_len)

x_index_list = create_x_index_list()

class RemoveResets(TransformationPass):
    def run(self, dag):
        for node in dag.op_nodes():
            if node.op.name == 'reset':
                
                dag.remove_op_node(node)
        return dag

#Function for NM optimizer
def run_circuit(phi):
    ar = AncillaRegister(2, 'ancilla')
    row_reg = QuantumRegister(N_L, 'l')
    col_reg = QuantumRegister(N_M, 'm')
    cr = ClassicalRegister(N_M + 2, 'cr')

    psi = QuantumCircuit(ar, col_reg, row_reg,  cr)


    estimated = np.copy(data)

    psi.h([x for x in range(QPU_len + 2)])
    createU_k(psi, estimated)

    psi.h(ar[0])

    createU_m(psi, col_reg, phi)
    psi.h(ar[1])
    psi.barrier([x for x in range(QPU_len + 2)])

    psi.h(col_reg)

    psi.measure(ar[0], cr[0])
    psi.measure(ar[1], cr[1])
    psi.barrier([x for x in range(QPU_len + 2)])
    for i in range(N_M):
        psi.measure(col_reg[i], cr[i + 2])

    # print(psi.decompose())
    # exit()

    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    isa_qc = pm.run(psi)

    with Session(backend=aer_sim) as session:
        sampler = Sampler(session=session)
        result = sampler.run([isa_qc], shots=shots).result()

    counts = result[0].data.cr.get_counts()
    return counts
    
def calc_expval(phi):
    expval = 1

    counts = run_circuit(phi)
    for x_list in x_index_list:
        expval += post_select(counts, x_list)
    expval /= math.pow(math.cos(phi[0]), 2)
    print(phi, expval)
    # exit()
    return expval

#The rest is basically for running the optimizer

init = [np.pi / 2]  * (2 ** N_M - 1)
init.insert(0, 3 * np.pi / 4, ) #Initial parameters
# init = [3.36172813, 1.57079631, 1.57079679, 0.22012573]
init = [np.pi, np.pi / 2, np.pi / 2, 0]
# init = [np.pi, np.pi, 0, 0]
bounds = [(-np.pi, np.pi)] * (2 ** N_M - 1)
bounds.insert(0, ( np.pi / 2, 3 * np.pi / 2))
bounds = tuple(bounds) #Bounds

res = optimize.minimize(fun = calc_expval, x0 = init, method = 'Nelder-Mead', options={'maxiter' : 200, 'disp': True}, bounds = bounds)
print(res.x)
print(res.message)


    #     if("{:0{width}b}".format(i, width=QPU_len) not in states): statesWithIndex.append(("{:0{width}b}".format(i, width=QPU_len), 0))
    #     service = QiskitRuntimeService(channel='ibm_quantum',token='8b7dbd957b8397e509a9b18af70f77a2853e7f9ef6a7cec345300e9e530f654061389b4ddfb86d9668cffe36379349f2d6c6d3a470609f927b423b66fac16254')

