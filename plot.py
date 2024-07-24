import matplotlib.pyplot as plt
import numpy as np

dataset_size = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

qc1_cx_count = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] #our
qc2_cx_count = [2, 16, 112, 384, 1280, 3584, 10240, 26624, 65536, 155648]  #naive
qc3_cx_count = [0, 2, 6, 14, 30, 62, 126, 254, 510, 1022] #built in

qc1_all_ops = [8, 13, 22, 39, 72, 137, 266, 523, 1036, 2061] #our
qc2_all_ops = [56, 42, 380, 1116, 4258, 12755, 35244, 90115, 218979, 515929] #naive
qc3_all_ops = [5, 56, 83, 138, 249, 472, 919, 1814, 3605, 7182] #built in

plt.yscale('log')
plt.plot(dataset_size, qc1_all_ops, label='phase_folding')
plt.plot(dataset_size, qc2_all_ops, label= 'qiskit_naive')
plt.plot(dataset_size, qc3_all_ops, ':', label='qiskit_built-in')
plt.legend(loc='best')
plt.xlabel("Dataset size")
plt.ylabel("Number of all elementary operations")
plt.show()