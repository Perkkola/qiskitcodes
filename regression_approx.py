from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, AncillaRegister
from qiskit.circuit import ParameterVector
import numpy as np

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

for i in range(len(data)):
       data[i] = np.square(data[i])


num_qubits = 5
num_layers = 8

qc = QuantumCircuit(5)
p =  ParameterVector('p', num_qubits * num_layers)

def create_ansatz(circuit):
    for i in range(num_layers):
        for j in range(num_qubits):
            circuit.ry(p[i * num_qubits + j], j)
        for j in range(0, (num_qubits - 1), 2):
            circuit.cx(j, j + 1)
        for j in range(1, num_qubits, 2):
            circuit.cx(j, j + 1)
        circuit.barrier([x for x in range(num_qubits)])
    circuit.measure_all()
    return circuit

print(create_ansatz(qc))
    
            