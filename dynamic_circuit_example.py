from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, CommutativeCancellation, Optimize1qGatesSimpleCommutation, TemplateOptimization, HoareOptimizer
from qiskit.circuit.library import HGate, XGate, ZGate, CXGate, CZGate
from qiskit.circuit import WhileLoopOp
# from qiskit import BasicAer, execute
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

mid_measurement = ClassicalRegister(1, name='mid')
final_measurement = ClassicalRegister(1, name='final')

control = QuantumRegister(1, name='control')
target = QuantumRegister(1, name='target')

circuit = QuantumCircuit(control, target, mid_measurement, final_measurement)

def trial(circuit, target, control, measure):
    circuit.h(control)
    circuit.cx(control, target)
    circuit.measure(control, measure)

trial(circuit, target, control, mid_measurement)
print(circuit)

max_trials = 64

for _ in range(max_trials - 1):
    with circuit.if_test((mid_measurement, 0b1)) as else_:
        pass
    with else_:
        trial(circuit, target, control, mid_measurement)


circuit.measure(control, mid_measurement)
circuit.measure(target, final_measurement)


print(circuit)

aer_sim = AerSimulator()
pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
isa_qc = pm.run(circuit)
print(len(isa_qc))
with Session(backend=aer_sim) as session:
    sampler = Sampler(session=session)
    result = sampler.run([isa_qc]).result()[0]
print(f">>> Simulator counts for mid: {result.data.mid.get_counts()}")
print(f">>> Simulator counts for final: {result.data.final.get_counts()}")
# qc = QuantumCircuit(qr, cr)

# with qc.while_loop((cr, 0)):
#     qc.h(0)
#     qc.measure(qr[0], cr[0])

# result = execute(qc, BasicAer.get_backend('qasm_simulator'), shots=2**9).result().get_counts()
# print(result)