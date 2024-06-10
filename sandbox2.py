from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
import numpy as np

reversed_number = int(("{:0{width}b}".format(3, width=5))[::-1], 2)
# print(reversed_number)

print(int(7 / 4), 7 % 4)


