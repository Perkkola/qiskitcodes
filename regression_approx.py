from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, AncillaRegister
from qiskit.circuit import ParameterVector
import numpy as np
from sklearn import metrics
from sympy import fwht
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from scipy import optimize
import multiprocessing

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
    data[i] = np.flip(np.append(X[i], y[i]))

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

data = data.flatten()

# data = [1/2, 1/2, 1/2, 1/2]

for i in range(len(data)):
    data[i] = np.square(data[i])

def arr_to_dict(arr):
    dist = {}
    for i in range(len(arr)):
        binary_string = ("{:0{width}b}".format(i, width=5))[::-1]
        dist[binary_string] = arr[i]
    return dist

def dict_to_arr(dic):
    return [x for x in dic.values()]

def create_ansatz(circuit):
    for i in range(num_layers):
        for j in range(num_qubits):
            circuit.ry(pv[i * num_qubits + j], j)
        for j in range(0, (num_qubits - num_qubits % 2), 2):
            circuit.cx(j, j + 1)
        for j in range(1, num_qubits - (num_qubits - 1) % 2, 2):
            circuit.cx(j, j + 1)
        circuit.barrier([x for x in range(num_qubits)])
    circuit.h(0)
    return circuit

def create_h_ansatz(circuit):
    for i in range(num_layers):
        for j in range(num_qubits):
            circuit.ry(pv[i * num_qubits + j], j)
        for j in range(0, (num_qubits - num_qubits % 2), 2):
            circuit.cx(j, j + 1)
        for j in range(1, num_qubits - (num_qubits - 1) % 2, 2):
            circuit.cx(j, j + 1)
        circuit.barrier([x for x in range(num_qubits)])
    circuit.h([x for x in range(num_qubits)])
    return circuit

def run_circ(circuit, parameters):
    circ = circuit.copy()
    b_qc = circ.assign_parameters({pv: parameters})
    b_qc.measure_all()

    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    isa_qc = pm.run(b_qc)

    sampler = Sampler(mode=aer_sim)
    result = sampler.run([isa_qc], shots=shots).result()

    counts = result[0].data.meas.get_counts()
    distribution = post_select(counts)
    return distribution

def post_select(counts):
    new_counts = {}
    discarded = 0
    for key in counts:
        if(key[-1] == '1'):
            new_key = key[:-1]
            if new_key in new_counts:
                new_counts[new_key] += counts[key]
            else:
                new_counts[new_key] = counts[key]
        else:
            discarded += counts[key]
    new_shots = shots - discarded
    distribution = {key: new_counts[key] / new_shots for key in new_counts.keys()}
    # dist_vals = np.fromiter(distribution.values(), dtype=float)
    return distribution

def kernel(j, k, gamma=1.0):
    x = -(j - k) ** 2 / (2 * gamma ** 2)
    return np.exp(x)

def mmd_rbf(X, Y):
    XX = sum([kernel(int(j[::-1], 2), int(k[::-1], 2)) * X[j] * X[k] for j in X.keys()
              for k in X.keys()])
    YY = sum([kernel(int(j[::-1], 2), int(k[::-1], 2)) * Y[j] * Y[k] for j in Y.keys()
              for k in Y.keys()])
    XY = sum([kernel(int(j[::-1], 2), int(k[::-1], 2)) * X[j] * Y[k] for j in X.keys()
              for k in Y.keys()])
    return XX + YY - 2 * XY

def loss(q, p, q_h, p_h):
    mmd = mmd_rbf(q, p)
    mmd_h = mmd_rbf(q_h, p_h)
    return (mmd + mmd_h) / 2

def gradient_block(q, p, q_h, p_h, parameters, s, index = None, return_dict = None):
    partial_diff_params = []
    for i in s:
        r_plus, r_minus = np.copy(parameters), np.copy(parameters)
        r_plus[i] = r_plus[i] + np.pi / 2
        r_minus[i] = r_minus[i] - np.pi / 2
        
        q_plus = run_circ(ansatz_qc, r_plus)
        q_minus = run_circ(ansatz_qc, r_minus)
        q_h_plus = arr_to_dict(fwht(dict_to_arr(q_plus)))
        q_h_minus = arr_to_dict(fwht(dict_to_arr(q_minus)))
        # q_h_plus = run_circ(h_ansatz_qc, r_plus)
        # q_h_minus = run_circ(h_ansatz_qc, r_minus)

        X_plusX = sum([kernel(int(j[::-1], 2), int(k[::-1], 2)) * q_plus[j] * q[k] for j in q_plus.keys()
                for k in q.keys()])
        X_minusX = sum([kernel(int(j[::-1], 2), int(k[::-1], 2)) * q_minus[j] * q[k] for j in q_minus.keys()
                for k in q.keys()])
        X_plusY = sum([kernel(int(j[::-1], 2), int(k[::-1], 2)) * q_plus[j] * p[k] for j in q_plus.keys()
                for k in p.keys()])
        X_minusY = sum([kernel(int(j[::-1], 2), int(k[::-1], 2)) * q_minus[j] * p[k] for j in q_minus.keys()
                for k in p.keys()])

        X_h_plusX_h = sum([kernel(int(j[::-1], 2), int(k[::-1], 2)) * q_h_plus[j] * q_h[k] for j in q_h_plus.keys()
                for k in q_h.keys()])
        X_h_minusX_h = sum([kernel(int(j[::-1], 2), int(k[::-1], 2)) * q_h_minus[j] * q_h[k] for j in q_h_minus.keys()
                for k in q_h.keys()])
        X_h_plusY_h = sum([kernel(int(j[::-1], 2), int(k[::-1], 2)) * q_h_plus[j] * p_h[k] for j in q_h_plus.keys()
                for k in p_h.keys()])
        X_h_minusY_h = sum([kernel(int(j[::-1], 2), int(k[::-1], 2)) * q_h_minus[j] * p_h[k] for j in q_h_minus.keys()
                for k in p_h.keys()])

        gradient = (X_plusX - X_minusX - X_plusY + X_minusY
                    + X_h_plusX_h - X_h_minusX_h - X_h_plusY_h + X_h_minusY_h) / 2
        partial_diff_params.append(gradient)
    if index != None: return_dict[index] = partial_diff_params
    else: return partial_diff_params

def gradient(q, p, q_h, p_h, parameters, parallelize=True):
    if parallelize:
        results = []
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        num_processes = 8
        jobs = []
        r = np.array([x for x in range(num_params)])
        s = np.array_split(r, num_processes)

        for i in range(num_processes):
            pr = multiprocessing.Process(target=gradient_block, args=(q, p, q_h, p_h, parameters, s[i], i, return_dict))
            jobs.append(pr)
            pr.start()
        for proc in jobs:
            proc.join()
        for i in range(num_processes):
            results.append(return_dict[i])
        results = np.array(results, dtype='float64').flatten()
        return results
    else:
        r = np.array([x for x in range(num_params)])
        diff_gradients = gradient_block(q, p, q_h, p_h, parameters, r)
        return np.array(diff_gradients, dtype='float64')

def gradient_descent(steps=50, lr=0.5):
    x = [np.pi / 2] * (num_params)

    # x = np.array(np.random.uniform(0, 2*np.pi, num_params))
    model_distributions = []
    loss_func_vals = []

    q = run_circ(ansatz_qc, x)
    p = arr_to_dict(data)
    # q_h = run_circ(h_ansatz_qc, x)

    q_h = arr_to_dict(fwht(dict_to_arr(q)))
    p_h = arr_to_dict(fwht(data))


    loss_val = loss(q, p, q_h, p_h)

    model_distributions.append(q)
    loss_func_vals.append(loss_val)

    print(f"Init loss value: {loss_val}")
    lowest = loss_val
    lowest_index = 0
    for i in range(steps):
        x -= lr*gradient(q, p, q_h, p_h, x, parallelize=True)
        q = run_circ(ansatz_qc, x)
        # q_h = run_circ(h_ansatz_qc, x)
        q_h = arr_to_dict(fwht(dict_to_arr(q)))
        loss_val = loss(q, p, q_h, p_h)

        if loss_val < lowest:
            lowest = loss_val
            lowest_index = i

        model_distributions.append(q)
        loss_func_vals.append(loss_val)

        print(f"Loss value: {loss_val}")
    
    print(f"Best loss value: {lowest}")
    print(f"Best model dist: {str(model_distributions[lowest_index])}")
    print(f"Target dist: {str(p)}")

num_qubits = 6
num_layers = 12
num_params = num_layers * num_qubits
shots = 2 ** 12
iteration = 0

model_distributions = []
loss_func_vals = []

qc = QuantumCircuit(num_qubits)
pv =  ParameterVector('p', num_params)

ansatz_qc = qc.copy()
h_ansatz_qc = qc.copy()

ansatz_qc = create_ansatz(ansatz_qc)
h_ansatz_qc = create_h_ansatz(h_ansatz_qc)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    gradient_descent()