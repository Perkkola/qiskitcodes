from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
import numpy as np
# Bell Circuit
qc = QuantumCircuit(1)
qc.rx(np.pi, 0)
qc.measure_all()
 
# Run the sampler job locally using AerSimulator.
# Session syntax is supported but ignored because local mode doesn't support sessions.
aer_sim = AerSimulator()
pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
isa_qc = pm.run(qc)
with Session(backend=aer_sim) as session:
    sampler = Sampler(session=session)
    result = sampler.run([isa_qc]).result()

print(result[0].data.meas.get_counts())

m = 3
n = 1

for i in range(2 ** m):
    binary_string = ("{:0{width}b}".format(i, width=m))[::-1]
    qc = QuantumCircuit(m)
    if i != 0: qc.h([i for i, x in enumerate(binary_string) if x == '1'])
    m_hat = qc.to_gate()
    print(m_hat)

    


