from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit import transpile
import numpy as np
from qiskit.circuit.library import RZGate, MCMT, RXGate, PhaseGate, CPhaseGate, CZGate
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis import synth_cnot_phase_aam
import sys
from gray_synth import synth_cnot_phase_aam
from functools import reduce

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

def create_x_index_list(qubit_length):
    x_index_list = []

    for i in range(1, 2 ** qubit_length):
        binary_string = ("{:0{width}b}".format(i, width=qubit_length))[::-1]
        indices = [i + 1 for i, x in enumerate(binary_string) if x == '1']
        x_index_list.append(indices)
    return x_index_list

def run_qc(shots, x_index_list):

    num_qubits = 3
    # qc = QuantumCircuit(num_qubits)
    # qc.x([1, 2])
    # ncrz_circ2(num_qubits, qc, np.pi /2, [x for x in range(1,  num_qubits)], 0)
    # print(qc)
    # print(transpile(qc, basis_gates=['cx', 'h', 'rz'], optimization_level=3))
    # print(Operator(transpile(qc, basis_gates=['cx', 'h', 'rz'], optimization_level=3).to_gate()).to_matrix())
    theta = 2 * np.pi
    cnots = [[1, 1],
             [0, 1]]
    thetas = [theta/2, -theta/2]

    thetas = generate_thetas(generate_thetas(generate_thetas(generate_thetas(thetas))))
    cnots = generate_cnots(generate_cnots(generate_cnots(generate_cnots(cnots))))
    # thetas_cp = np.copy(thetas)
    x_lsit = [1, 2, 3, 4, 5]
    thetas = flip_signs(cnots, thetas, x_lsit)
    thetas = thetas.tolist()
    # x_index_lists = create_x_index_list(num_qubits - 1)[::-1]

    # thetas_sum = np.copy(thetas)
    # for x_index_list in x_index_lists:
    #     flipped_thetas = flip_signs(cnots, thetas_cp, x_index_list)
    #     # all_thetas.append(flipped_thetas)
    #     thetas_sum = np.array(thetas_sum) + np.array(flipped_thetas)
    # thetas_sum = thetas_sum.tolist()
    # all_thetas.append(thetas_cp)
    # print(phase_sum_dict)
    # print(x_index_lists)
    # print(np.array(cnots))
    # print(thetas)
    # exit()
    # thetas[0] *= -1
    # thetas[2] *= -1
    # thetas[2] *= -1
    # thetas[3] *= -1
    # thetas[7] *= -1
    # print(thetas)
    # print(np.array(cnots))
    # exit()
    qc = QuantumCircuit(7)
    # qc.h([x for x in range(3)])
    gray_qc = synth_cnot_phase_aam(cnots, thetas)
    qc.append(gray_qc.to_gate(), [0, 2, 3, 4, 5, 6])
    print(qc.decompose())
    # exit()
    print(np.array(Operator(transpile(qc, basis_gates=['cx', 'h', 'x', 'rz'], optimization_level=1).to_gate()).to_matrix()))
    # exit()
    print('///////////')






    qc2 = QuantumCircuit(7)
    rz = RZGate(2 * np.pi)
    mcrz = MCMT(rz, 5, 1)

    # qc2.h([x for x in range(3)])
    qc2.x([2, 3, 4, 5, 6])
    qc2.append(mcrz, [6, 5, 4, 3, 2, 0])
    qc2.x([2, 3, 4, 5, 6])

    # qc2.x([2])
    # qc2.append(mcrz, [2, 1, 0])
    # qc2.x([2])

    # qc2.x([1])
    # qc2.append(mcrz, [2, 1, 0])
    # qc2.x([1])

    # qc2.append(mcrz, [2, 1, 0])
    print(transpile(qc2, basis_gates=['cx', 'h', 'x', 'rz'], optimization_level=3))
    print(np.array(Operator(transpile(qc2, basis_gates=['cx', 'h', 'x', 'rz'], optimization_level=1).to_gate()).to_matrix()))
    exit()
    print('///////////')
    qc3 = QuantumCircuit(2)
    qc3.cx(1, 0)
    qc3.rz(np.pi, 0)
    qc3.cx(1, 0)
    qc3.rz(-np.pi, 0)
    qc3.rz(-np.pi, 1)
    print(qc3)
    # qc3.cz(1, 0)
    print(Operator(transpile(qc3, basis_gates=['cx', 'h', 'rz'], optimization_level=1).to_gate()).to_matrix())
    exit()
    # qc.mcp(0.25, [3, 2, 1], 0)

    # qc.h(qr)
    # qc.initialize([1/(2*np.sqrt(2)), 1/2, 1/(2*np.sqrt(2)), 1/np.sqrt(2)], qc.qubits)
    # qc.h(x_index_list)
    qc.measure_all()

    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    isa_qc = pm.run(qc)
    # print(qc.decompose(reps=2))

    with Session(backend=aer_sim) as session:
        sampler = Sampler(session=session)
        result = sampler.run([isa_qc], shots=shots).result()

    # with Session(backend=aer_sim) as session:
    #     sampler = Sampler(session=session)
    #     result = sampler.run([isa_qc], shots=2**16).result()
    results = result[0].data.meas.get_counts()
    print(results)
    exit()
    return results

shots = 2**16

def cut_counts(counts, bit_indexes):
    bit_indexes.sort(reverse=True) 
    new_counts = {}

    for key in counts:
        new_key = ''
        for index in bit_indexes:
            new_key += key[-1 - index]
        if new_key in new_counts:
            new_counts[new_key] += counts[key]
        else:
            new_counts[new_key] = counts[key]

    return new_counts

def post_select(counts, z_index_list):

    
    x_counts = cut_counts(counts, z_index_list)
    expval = 0

    for key, value in zip(x_counts.keys(), x_counts.values()):
        if(key.count('1') % 2 == 0):
            expval += value
        else:
            expval -= value
    return expval
# print(post_select(counts) / shots)
indices = z_index_list = [[0], [1], [0, 1]]
expval = 1
counts = run_qc(shots, [0, 1])
for li in indices:
    expval += post_select(counts, li) / shots
print(expval)
# for key, value in zip(counts.keys(), counts.values()):
#     print(key[:-2])