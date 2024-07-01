from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit import transpile
import numpy as np
from qiskit.circuit.library import RZGate, MCMT

def run_qc(shots, x_index_list):

    cl = ClassicalRegister(4, 'cl1')

    qr = QuantumRegister(4, 'qr')

    qc = QuantumCircuit(qr, cl)
    rz = RZGate(0.25)
    mcrz = MCMT(rz, 3, 1)
    qc.append(mcrz, [3, 2, 1, 0])
    # qc.mcp(0.25, [3, 2, 1], 0)
    print(transpile(qc, basis_gates=['cx', 'h', 'rz'], optimization_level=3))
    exit()
    # qc.h(qr)
    qc.initialize([1/(2*np.sqrt(2)), 1/2, 1/(2*np.sqrt(2)), 1/np.sqrt(2)], qc.qubits)
    qc.h(x_index_list)
    qc.measure_all()

    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    isa_qc = pm.run(qc)
    print(qc)

    with Session(backend=aer_sim) as session:
        sampler = Sampler(session=session)
        result = sampler.run([isa_qc], shots=shots).result()

    # with Session(backend=aer_sim) as session:
    #     sampler = Sampler(session=session)
    #     result = sampler.run([isa_qc], shots=2**16).result()


    return result[0].data.meas.get_counts()

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