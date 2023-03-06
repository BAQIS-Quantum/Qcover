# from quafu import User
# user = User()
# user.save_apitoken('csEFB6U-Kn5f-x5Dh3kk2DX6eEf4JqAaI20TmP-YHAO.0nM2IDOxQDN2YTM6ICc4VmIsIyMyEjI6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye')

import numpy as np
from quafu import QuantumCircuit
import matplotlib.pyplot as plt

q = QuantumCircuit(5)
q.x(0)
q.x(1)
q.cnot(2, 1)
q.ry(1, np.pi/2)
q.rx(2, np.pi)
q.rz(3, 0.1)
q.cz(2, 3)
measures = [0, 1, 2, 3]
cbits = [0, 1, 2, 3]
q.measure(measures,  cbits=cbits)
q.draw_circuit(width=4)

qc = QuantumCircuit(4)
test_ghz = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
cx q[0],q[1];
cx q[0],q[2];
cx q[0],q[3];
"""
qc.from_openqasm(test_ghz)
qc.draw_circuit()

from quafu import Task
task = Task()
task.load_account()
task.config(backend="ScQ-P10", shots=2000, compile=True)
res = task.send(q)

print(res.counts) #counts
print(res.amplitudes) #amplitude
res.plot_amplitudes()

from quafu import simulate
simu_res = simulate(q, output="amplitudes")
simu_res.plot_amplitudes(full=True)

res = task.send(qc)
res.plot_amplitudes()

simu_res = simulate(qc)
simu_res.plot_amplitudes(full=True)
plt.show()