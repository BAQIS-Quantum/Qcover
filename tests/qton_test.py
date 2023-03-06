import time
from Qcover import COBYLA
from Qcover.core import Qcover
from Qcover.applications import MaxCut

p = 1
node_num = [10, 50, 100, 500, 1000]
node_d = [3, 4, 5, 6]
for i in node_num:
    pt, st = [], []
    pv, sv = [], []
    for nd in node_d:
        mxt = MaxCut(node_num=i, node_degree=nd)
        g, shift = mxt.run()
        from Qcover.backends import CircuitByQton
        qt = CircuitByQton(expectation_calc_method="tensor")  #
        optc = COBYLA(options={'tol': 1e-3, 'disp': True})
        qct = Qcover(g, p,
                    optimizer=optc,
                    backend=qt)  # qiskit_bc, ,qulacs_bc

        t1 = time.time()
        sol = qct.run(is_parallel=False, mode='QAQA')  # True
        t2 = time.time()
        st.append(t2 - t1)

        t1 = time.time()
        sol = qct.run(is_parallel=True, mode='QAQA')  #
        t2 = time.time()
        pt.append(t2 - t1)

        qcv = Qcover(g, p,
                    optimizer=COBYLA(options={'tol': 1e-3, 'disp': True}),
                    backend=CircuitByQton())  # qiskit_bc, ,qulacs_bc
        t1 = time.time()
        sol = qcv.run(is_parallel=False, mode='QAQA')  # True
        t2 = time.time()
        sv.append(t2 - t1)

        t1 = time.time()
        sol = qcv.run(is_parallel=True, mode='QAQA')  #
        t2 = time.time()
        pv.append(t2 - t1)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(node_d, pt, "ob-", label="parallel tensor")
plt.plot(node_d, st, "^r-", label="serial tensor")
plt.plot(node_d, pv, "*g-", label="parallel statevector")
plt.plot(node_d, sv, "dy-", label="serial statevector")
plt.ylabel('Time cost')
plt.xlabel('node degree')
plt.title("node is %d" % i)
plt.legend()
plt.savefig('/public/home/humengjun/Qcover/result_log/qton_tensor/%d.png' % i)
plt.close('all')