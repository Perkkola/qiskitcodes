from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import MCMT, RZGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import optimize
from functools import reduce

s = np.array([1, 2, 3])
v = np.array([1, 2 ,3])

bit_list = [1, 0, 1]
x_index_list = [1, 2, 3]
print(bit_list[:-1])
exit()

print([bit_list[x] for x in x_index_list])
print(reduce(lambda a, b: a ^ b, bit_list))


exit()

p = ParameterVector('p', 9)

qc = QuantumCircuit(3)
qc.rx(p[0], 0)
qc.rx(p[1], 1)
qc.rx(p[2], 2)
qc.ry(p[3], 0)
qc.ry(p[4], 1)
qc.ry(p[5], 2)
qc.rz(p[6], 0)
qc.rz(p[7], 1)
qc.rz(p[8], 2)


shots = 2 ** 16
def run_circ(params):
    b_qc = qc.assign_parameters({p: params})
    b_qc.measure_all()


    aer_sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    isa_qc = pm.run(b_qc)

    
    sampler = Sampler(mode=aer_sim)
    result = sampler.run([isa_qc], shots=shots).result()

    counts = result[0].data.meas.get_counts()
    return counts


def opt(params):
    counts = run_circ(params)
    probs = []
    for value in counts.values():
            probs.append(value / shots)
    
    mse = mean_squared_error(data, probs)
    print(f"Params: {params}")
    print(f"Probs: {probs}")
    print(f"MSE: {mse}")
    return mse


init = [np.pi / 2]  * 9

bounds = [(-np.pi, np.pi)] * (9)
bounds = tuple(bounds) #Bounds

res = optimize.minimize(fun = opt, x0 = init, method = 'CG', options={'maxiter' : 300, 'disp': True}, bounds = bounds)
print(res.x)
print(res.message)