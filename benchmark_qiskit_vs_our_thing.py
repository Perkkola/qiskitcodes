# Prepare input data
import random
import os
import math
import numpy as np
import pandas as pd
import numpy.random as npr
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
    thetas = np.copy(thetas)
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




def create_circ_qiskit(circuit, data_arr):
    data_arr = np.copy(data_arr)
    for i in range(len(data_arr)):
        bit_string = ("{:0{width}b}".format(i, width=QPU_len))[::-1]
        x_index_list = [i + 1 for i, x in enumerate(bit_string) if x == '0']
        mcrz_index_list = [x + 1 for x in range(QPU_len)]
        mcrz_index_list.append(0)
        for j in range(2):
            if len(x_index_list) > 0: circuit.x(x_index_list)
            if j == 0:
                rz = RZGate(data[i] * 2)
                mcrz = MCMT(rz, QPU_len, 1)
                circuit.append(mcrz, mcrz_index_list)

        # circuit.barrier([x for x in range(QPU_len + 1)])

def create_circ_phase_folding(circuit, data_arr):
    data_arr = np.copy(data_arr)
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

    for thetas_cp, x_phase_index_list in zip(all_thetas[:-1], x_phase_index_lists):
        flipped_thetas = flip_signs(cnots, np.copy(thetas_cp), x_phase_index_list)
        thetas = np.array(thetas) + np.array(flipped_thetas)
    thetas = thetas.tolist()

    gray_qc = synth_cnot_phase_aam(cnots, thetas)
    circuit.append(gray_qc.to_gate(), [x for x in range(QPU_len + 1)])
    # circuit.barrier([x for x in range(QPU_len + 1)])

for i in range(1, 11):
    QPU_len = i

    data = npr.uniform(0, 2*np.pi, 2 ** QPU_len)

    # ar1 = AncillaRegister(1, 'ancilla1')
    # qpu1 = QuantumRegister(QPU_len, 'QPU1')
    # cl1 = ClassicalRegister(1, 'cl1')
    # qc1 = QuantumCircuit(ar1, qpu1, cl1)

    ar2 = AncillaRegister(1, 'ancilla2')
    qpu2 = QuantumRegister(QPU_len, 'QPU2')
    cl2 = ClassicalRegister(1, 'cl2')
    qc2 = QuantumCircuit(ar2, qpu2, cl2)

    # qpu3 = QuantumRegister(QPU_len)
    # qc3 = QuantumCircuit(qpu3)

    # qc1.h([x for x in range(QPU_len + 1)])
    qc2.h([x for x in range(QPU_len + 1)])

    estimated = np.copy(data)

    # create_circ_phase_folding(qc1, estimated)
    # qc3.prepare_state(estimated, qpu3, normalize=True)
    create_circ_qiskit(qc2, estimated)

    # qc1.h(ar1[0])
    qc2.h(ar2[0])

    # qc1.measure(ar1[0], cl1[0])
    qc2.measure(ar2[0], cl2[0])

    # qc1 = transpile(qc1, basis_gates=['cx', 'h', 'x', 'rz'], optimization_level=3)
    qc2 = transpile(qc2, basis_gates=['cx', 'h', 'x', 'rz'], optimization_level=3)
    # qc3 = transpile(qc3, basis_gates=['cx', 'h', 'x', 'rz'], optimization_level=3)


    # qc1_ops = qc1.count_ops()
    qc2_ops = qc2.count_ops()
    # qc3_ops = qc3.count_ops()
    print(qc2_ops)
    # qc1_cx = qc1_ops['cx']
    # qc2_cx = qc2_ops['cx']
    qc2_cx = qc2_ops['cx'] if 'cx' in qc2_ops else 0
    qc2_rz = qc2_ops['rz'] if 'rz' in qc2_ops else 0 
    qc2_h = qc2_ops['h'] if 'h' in qc2_ops else 0
    qc2_x = qc2_ops['x'] if 'x' in qc2_ops else 0
    qc2_meas = qc2_ops['measure'] if 'measure' in qc2_ops else 0

    # qc1_all = qc1_ops['cx'] + qc1_ops['rz'] + qc1_ops['h'] + qc1_ops['measure']
    # qc2_all = qc2_ops['cx'] + qc2_ops['rz'] + qc2_ops['h'] + qc2_ops['measure']
    # qc3_all = qc3_cx + qc3_h + qc3_rz + qc3_x
    qc2_all = qc2_cx + qc2_h + qc2_rz + qc2_x
    
    print(f"qc2 cx count: {qc2_cx}, all ops: {qc2_all} for dataset size: {2 ** QPU_len}")
    # print(f"qc2 cx count: {qc2_cx}, all ops: {qc2_all} for dataset size: {2 ** QPU_len}")
