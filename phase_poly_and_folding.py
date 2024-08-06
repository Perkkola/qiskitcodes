# Prepare input data
import random
import os
import math
import sys
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, AncillaRegister
from qiskit.transpiler import PassManager, CouplingMap, TransformationPass
from qiskit.circuit.library import RZGate, MCMT, RXGate
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit.quantum_info import Statevector
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator
from scipy import optimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from gray_synth import synth_cnot_phase_aam

from pauliopt.pauli import synthesis# from iqm.qiskit_iqm import IQMProvider


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