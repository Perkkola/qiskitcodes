from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler

ar = QuantumRegister(1, 'ar')

cl = ClassicalRegister(3, 'cl1')

qr = QuantumRegister(1, 'qr')

qc = QuantumCircuit(ar, qr, cl)

# qc.x(ar[0])
qc.x(qr[0])
qc.measure(ar[0], cl[0])
qc.reset(ar[0])
qc.measure(ar[0], cl[1])
qc.measure(qr[0], cl[2])


aer_sim = AerSimulator()
pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
isa_qc = pm.run(qc)
print(isa_qc)
with Session(backend=aer_sim) as session:
    sampler = Sampler(session=session)
    result = sampler.run([isa_qc]).result()


cl1c = result[0].data.cl1.get_counts()


print(cl1c)

# for key, value in zip(counts.keys(), counts.values()):
#     print(key[:-2])