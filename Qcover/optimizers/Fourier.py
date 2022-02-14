import sys
import numpy as np
from typing import Optional
from scipy import optimize as opt
from math import *
from Qcover.optimizers import Optimizer
from Qcover.exceptions import ArrayShapeError


class Fourier(Optimizer):
    """
        Fourier optimizer: a heuristic optimization method for QAOA,
        implemented according to the paper
        "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices"
    """
    def __init__(self,
                 p: int = 1,
                 q: Optional[int] = None,   # 4
                 r: Optional[int] = 0,
                 alpha: Optional[float] = 0.6,
                 optimize_method: Optional[str] = 'COBYLA',
                 options: dict = None, #{'maxiter':300, 'disp':True, 'rhobeg': 1.0, 'tol':1e-6},
                 initial_point: Optional[list] = None
                 ) -> None:
        """
        initialize a optimizer of FOURIER with parameters q and R
        Args:
            p: the parameter in QAOA paper
            q: the maximum frequency component allowed in the amplitude parameters <⃗u, ⃗v>
            r: the number of random perturbations to add
            alpha:
        """
        super().__init__()
        self._p = p
        self._q = q if q is not None and (q < self._p and q >= 1) else self._p
        self._r = r
        self._alpha = alpha
        self._optimize_method = optimize_method
        self._options = options
        self._initial_point = initial_point # used to initialize (u, v) that shape are defined by q

        self._objective_function = None

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, aq):
        self._q = aq if (aq < self._p and aq >= 1) else self._p

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, ar):
        self._r = ar

    def calculate_gb(self, step, pargs):
        upb = min(self._q, step)
        u, v = pargs[:upb], pargs[upb:]
        gamma, beta = np.zeros(step), np.zeros(step)
        for i in range(1, step + 1):
            for k in range(1, upb + 1):
                gamma[i - 1] += u[k - 1] * sin((k - 0.5) * (i - 0.5) * pi / step)
                beta[i - 1] += v[k - 1] * cos((k - 0.5) * (i - 0.5) * pi / step)

            if gamma[i - 1] < -np.pi / 2:
                gamma[i - 1] = -np.pi / 2 + 0.01
            if beta[i - 1] < -np.pi / 4:
                beta[i - 1] = -np.pi / 4 + 0.01
            if gamma[i - 1] > np.pi / 2:
                gamma[i - 1] = np.pi / 2 - 0.01
            if beta[i - 1] > np.pi / 4:
                beta[i - 1] = np.pi / 4 - 0.01
        return gamma, beta

    def loss_function(self, pargs, step):
        gamma, beta = self.calculate_gb(step, pargs)
        return self._objective_function(np.append(gamma, beta), step)

    def _minimize(self, objective_function):
        """
        minimize the loss function
        Args:
            loss: the loss function

        Returns:
            x: the optimized gamma and beta
            value: the optimized value of loss function
            nfev: is the number of objective function calls
        """
        self._objective_function = objective_function

        nfev = 0
        ul, vl = None, None
        u_best, v_best = None, None
        for j in range(1, self._p + 1):
            u_list, v_list = [], []
            min_val = float("inf")
            if j == 1:
                # Attention:
                # though only used in j=1, but to be consistent with other optimizers,
                # self._initial_point should be defined according to q,
                # which is different with other optimizes
                ul, vl = list(self._initial_point[: j]), list(self._initial_point[self._q: self._q+j])
            else:
                if j <= self._q:
                    ul.append(0)
                    vl.append(0)

                for r in range(self._r + 1):
                    u_nx, v_nx = u_best.copy(), v_best.copy()
                    if r > 0:
                        for i, _ in enumerate(u_best):
                            u_nx[i] = u_nx[i] + self._alpha * np.random.normal(loc=0, scale=fabs(u_best[i]))
                            v_nx[i] = v_nx[i] + self._alpha * np.random.normal(loc=0, scale=fabs(v_best[i]))
                    if j <= self._q:
                        u_nx.append(0)
                        v_nx.append(0)
                    u_list.append(u_nx)
                    v_list.append(v_nx)

            for idx in range(len(u_list) + 1):
                if idx == 0:
                    u_cal, v_cal = ul, vl
                else:
                    u_cal, v_cal = u_list[idx - 1], v_list[idx - 1]

                res = opt.minimize(self.loss_function,
                                    x0=np.append(u_cal, v_cal),
                                    args=j,
                                    method=self._optimize_method,
                                    jac=opt.rosen_der,
                                    options=self._options)

                upb = min(self._q, j)
                if idx == 0:
                    ul, vl = list(res["x"][:upb]), list(res["x"][upb:])

                func_val = res["fun"]
                if func_val < min_val:
                    min_val = func_val
                    u_best, v_best = list(res["x"][:upb]), list(res["x"][upb:])

                nfev += res["nfev"]

        gamma_list, beta_list = self.calculate_gb(self._p, u_best + v_best)
        return np.append(gamma_list, beta_list), min_val, nfev
        # return {"gamma": gamma_list, "beta": beta_list, "optimal value": min_val, "nfev": nfev}

    def optimize(self, objective_function):
        if self._initial_point is None:
            self._initial_point = np.array([np.random.random() for x in range(2 * self._q)])
        else:
            try:
                if len(self._initial_point) != 2 * self._q:
                    raise ArrayShapeError("The shape of initial parameters is not match with q")
            except ArrayShapeError as e:
                print(e)
                sys.exit()

        return self._minimize(objective_function)
