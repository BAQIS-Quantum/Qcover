# import sys
# sys.path.append('/public/home/humengjun/Qcover/')
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tests.core_without_RQAOA import Qcover
from Qcover.applications.max_cut import MaxCut
from Qcover.optimizers import COBYLA, Fourier
from Qcover.backends import CircuitByQulacs, CircuitByTensor

node_num = 23  # even
node_degree = [2, 4, 6, 8]
plist = [4, 8, 12, 16]


ts_bc = CircuitByTensor()
qulacs_bc = CircuitByQulacs()

for d in node_degree:
    ct, gt, it, ft = [], [], [], []
    ce, ge, ie, fe = [], [], [], []
    mxt = MaxCut(node_num=node_num, node_degree=d)
    ising_g = mxt.run()
    for p in plist:
        print("node degree is %s, p value is %s" %(str(d), str(p)))
        print("runing Fourier")
        optf = Fourier(p=p, q=4, r=1, alpha=0.6, optimize_method="COBYLA", options={'tol': 1e-3, 'disp':False})
        qf = Qcover(ising_g, p=p,
                    optimizer=optf,
                    backend=CircuitByQulacs())  #qulacs_bc ts_bc

        time_start = time.time()
        res = qf.run(is_parallel=False)  # True
        time_end = time.time()
        ft.append(time_end - time_start)
        fe.append(res["Expectation of Hamiltonian"])
        nfev = res["Total iterations"]
        print("time cost by Fourier is: ", time_end - time_start)

        print("--------------------------------------------")
        print("runing COBYLA")
        optc = COBYLA(options={'maxiter': nfev, 'tol': 1e-3, 'disp': True})
        qc = Qcover(ising_g, p=p,
                    optimizer=optc,
                    backend=CircuitByQulacs())  # ts_bc

        time_start = time.time()
        res = qc.run(is_parallel=False)  # True
        time_end = time.time()
        ct.append(time_end - time_start)
        ce.append(res["Expectation of Hamiltonian"])
        print("time cost by COBYLA is: ", time_end - time_start)

        # print("--------------------------------------------")
        # print("runing GradientDescent")
        # optg = GradientDescent(maxiter=nfev, tol=1e-6)
        # qg = Qcover(ising_g, p=p,
        #             optimizer=optg,
        #             backend=qulacs_bc)  # ts_bc
        #
        # time_start = time.time()
        # res = qg.run(is_parallel=False)  # True
        # time_end = time.time()
        # gt.append(time_end - time_start)
        # print("time cost by GradientDescent is: ", time_end - time_start)
        #
        # print("--------------------------------------------")
        # print("runing Intep")
        # opti = Interp(optimize_method="COBYLA")  #
        # qi = Qcover(ising_g, p=p,
        #             optimizer=opti,
        #             backend=qulacs_bc)  # ts_bc
        #
        # time_start = time.time()
        # res = qi.run(is_parallel=False)  # True
        # time_end = time.time()
        # it.append(time_end - time_start)
        # print("time cost by Interp is: ", time_end - time_start)

    plt.figure(1)
    plt.subplot(121)
    plt.plot(plist[:len(ft)], ft, "ob-", label="Fourier degree=%s" % str(d))
    plt.plot(plist[:len(ct)], ct, "^r-", label="COBYLA degree=%s" % str(d))
    # plt.plot(plist[:len(gt)], gt, "*g-", label="GradientDescent %s" % str(d))
    # plt.plot(plist[:len(it)], it, "xy-", label="Interp %s" % str(d))
    plt.ylabel('time cost(s)')
    plt.xlabel('P value')
    plt.legend()
    plt.subplot(122)
    plt.plot(plist[:len(fe)], fe, "ob-", label="Fourier degree=%s" % str(d))
    plt.plot(plist[:len(ce)], ce, "^r-", label="COBYLA degree=%s" % str(d))
    plt.ylabel('expectation')
    plt.xlabel('P value')
    # plt.savefig('/home/wfzhuang/data/Qcover/result_log/backends_compare/res_serial_%s.png' % str(d))
    plt.savefig('/public/home/humengjun/Qcover/result_log/fourier_node23/ptm_r1_%s.png' % str(d))   #fourier_test
    # plt.savefig('/home/puyanan/QCover/result_log/backends_compare/tm_serial_%s.png' % str(d))
    # plt.savefig('E:/Working_projects/QAOA/QCover/result_log/backends_compare/res_serial_%s.png' % str(d))  # maxcut_serial
    # plt.show()
    plt.cla()
