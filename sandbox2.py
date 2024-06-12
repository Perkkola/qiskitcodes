from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import MCMT, RZGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
import numpy as np

qc = QuantumCircuit(3)
# qc.mcp(0.25, [0, 1], 2)
rz = RZGate(0.5)
cx = MCMT(rz, 2, 1)
qc.append(cx, [2, 1, 0])
# print(qc.decompose().decompose().decompose().decompose())
# qc.barrier()

print(qc.decompose())
print([x for x in range(5)][::-1])