from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
import numpy as np

qc = QuantumCircuit(3)
qc.mcp(0.25, [0, 1], 2)
print(qc.decompose().decompose().decompose().decompose())
qc.barrier()
qc.p()