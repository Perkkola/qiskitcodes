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
np.set_printoptions(threshold=sys.maxsize)

def crz_circ(circ, theta, control, target):
    circ.rz(theta, target)
    circ.cx(control, target)
    circ.rz(-theta, target)
    circ.cx(control, target)

    # circ.cx(control, target)
    # circ.rz(theta, target)
    # circ.cx(control, target)
    # circ.rz(-theta, target)
    # circ.rz(-theta, control)
    
    # circ.p(theta, control)
    # circ.cx(control, target)
    # circ.p(-theta, target)
    # circ.cx(control, target)
    # circ.p(theta, target)



def ncrz_circ(n_qubits, circ, theta, controls, target, iteration = 1):
    if (len(controls) == 1):
        crz_circ(circ, theta / 2, controls[0], target)
    else:
        ncrz_circ(n_qubits, circ, theta / 2, controls[1:], target + 1, iteration + 1)
        # circ.barrier([x for x in range(n_qubits)])
        ncrz_circ(n_qubits, circ, theta / 2, controls[1:], target, iteration + 1)
        # circ.barrier([x for x in range(n_qubits)])
        circ.cx(iteration, target)
        # circ.cx(1, 0)
        ncrz_circ(n_qubits, circ, -theta / 2, controls[1:], target, iteration + 1)
        # circ.barrier([x for x in range(n_qubits)])
        circ.cx(iteration, target)
        # circ.cx(1, 0)

def ncrz_circ2(n_qubits, circ, theta, controls, target, iteration = 0, long = False):
    if (len(controls) == 1):
        crz_circ(circ, theta / 2, controls[0], target)
    else:
        ncrz_circ2(n_qubits, circ, theta / 2, controls[:-1], target, iteration + 1)
        circ.cx(controls[-1], controls[-2])
        # circ.barrier([x for x in range(n_qubits)])


        ncrz_circ2(n_qubits, circ, -theta / 2, controls[:-1], target, iteration + 1)
        circ.cx(controls[-1], controls[-2])
        # circ.barrier([x for x in range(n_qubits)])
        # circ.cx(1, 0)
        ncrz_circ2(n_qubits, circ, theta / 2, controls[1:], target, iteration + 1, True)
        # circ.barrier([x for x in range(n_qubits)])
        # circ.cx(iteration, target)

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

def run_qc(shots, x_index_list):

    # num_qubits = 3
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

    thetas = generate_thetas(generate_thetas(thetas))
    cnots = generate_cnots(generate_cnots(cnots))
    print(np.array(cnots))
    # thetas[0] *= -1
    # thetas[2] *= -1
    # thetas[2] *= -1
    # thetas[3] *= -1
    # thetas[7] *= -1
    # print(thetas)
    # print(np.array(cnots))
    # exit()
    qc = QuantumCircuit(4)
    # qc.x([1, 2])
    gray_qc = synth_cnot_phase_aam(cnots, thetas)
    # print(gray_qc)
    qc.append(gray_qc.to_gate(), [0 ,1, 2, 3])
    # qc.x([1, 2])
    print(qc.decompose())
    # exit()
    print(Operator(transpile(qc, basis_gates=['cx', 'h', 'rz'], optimization_level=1).to_gate()).to_matrix())
    exit()
    print('///////////')
    qc2 = QuantumCircuit(3)
    # qc2.x([1,2])
    # rz = PhaseGate(np.pi/8)
    rz = RZGate(2 * np.pi)
    mcrz = MCMT(rz, 2, 1)
    qc2.append(mcrz, [2, 1, 0])
    # print(qc2)
    # print(transpile(qc2, basis_gates=['cx', 'h', 'rz'], optimization_level=3))
    # exit()
    print(Operator(transpile(qc2, basis_gates=['cx', 'h', 'rz'], optimization_level=1).to_gate()).to_matrix())
    # qc3 = QuantumCircuit(2)
    # qc3.cz(1, 0)
    # print(Operator(transpile(qc3, basis_gates=['cx', 'h', 'rz'], optimization_level=1).to_gate()).to_matrix())
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