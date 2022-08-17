import sys
import numpy as np
from typing import Optional
from scipy import optimize as opt
from Qcover.optimizers import Optimizer
from Qcover.exceptions import ArrayShapeError, OptimizerConfigError


class Interp(Optimizer):
    """
        Interp optimizer: a heuristic optimization method for QAOA,
        implemented according to the paper
        "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices"
    """
    def __init__(self,
                 optimize_method='COBYLA',
                 options: dict = None, #{'maxiter':300, 'disp':True, 'rhobeg': 1.0, 'tol':1e-6},
                 initial_point: Optional[np.ndarray] = None):
        super().__init__()
        self._p = None
        self._optimize_method = optimize_method
        self._options = options
        self._initial_point = initial_point

    def _minimize(self, objective_function):
        """
        minimize the loss function
        Args:
            loss: the loss function
            initial_point: the init parameters of gamma and beta

        Returns:
            x: the optimized gamma and beta
            value: the optimized value of loss function
            nfev: is the number of objective function calls
        """
        nfev = 0
        for k in range(1, self._p + 1):
            if k == 1:
                # though only used in p=1, but to be consistent with other optimizers,
                # self._initial_point should be defined according to p
                gamma_list, beta_list = self._initial_point[: k], self._initial_point[self._p: self._p + k]
                gamma_list = np.insert(gamma_list, 0, 0)
                beta_list = np.insert(beta_list, 0, 0)
            else:
                gamma_list = np.insert(gamma_list, 0, 0)
                beta_list = np.insert(beta_list, 0, 0)
                gamma_list = np.append(gamma_list, 0)
                beta_list = np.append(beta_list, 0)
                gamma_list_new, beta_list_new = gamma_list, beta_list
                for i in range(1, k + 1):
                    gamma_list_new[i] = (i - 1) / k * gamma_list[i - 1] + (k - i + 1) / k * gamma_list[i]
                    beta_list_new[i] = (i - 1) / k * beta_list[i - 1] + (k - i + 1) / k * beta_list[i]

                    if gamma_list_new[i] < -np.pi / 2:
                        gamma_list_new[i] = -np.pi / 2 + 0.01
                    if beta_list_new[i] < -np.pi / 4:
                        beta_list_new[i] = -np.pi / 4 + 0.01
                    if gamma_list_new[i] > np.pi / 2:
                        gamma_list_new[i] = np.pi / 2 - 0.01
                    if beta_list_new[i] > np.pi / 4:
                        beta_list_new[i] = np.pi / 4 - 0.01

                gamma_list, beta_list = gamma_list_new, beta_list_new

            res = opt.minimize(objective_function,
                               x0=np.append(gamma_list[1:k+1], beta_list[1:k+1]),
                               args=k,
                               method=self._optimize_method,
                               jac=opt.rosen_der,
                               options=self._options)

            gamma_list, beta_list = res["x"][:k], res["x"][k:]
            value = res["fun"]
            nfev += res["nfev"]

            try:
                optv_f, aprv_f = "optimal_value" in self._options, "approximate_ratio" in self._options
                if optv_f ^ aprv_f:
                    raise OptimizerConfigError("optimal value and approximate ratio option should be set simultaneously to the optimizer")
                if optv_f & aprv_f:
                    optv, aprv = self._options["optimal_value"], self._options["approximate_ratio"]
                    if optv * aprv >= value:
                        print("The target approximation relative to the optimal value is reached. "
                              "The value of P is", k)
                        break
            except OptimizerConfigError as e:
                print(e)
                sys.exit()

        return np.append(gamma_list, beta_list), value, nfev
        # return {"gamma": gamma_list, "beta": beta_list, "optimal value": value, "nfev": nfev}

    def optimize(self, objective_function):
        if self._initial_point is None:
            self._initial_point = np.array([np.random.random() for x in range(2 * self._p)])
        else:
            try:
                if len(self._initial_point) != 2 * self._p:
                    raise ArrayShapeError("The shape of initial parameters is not match with p")
            except ArrayShapeError as e:
                print(e)
                sys.exit()

        return self._minimize(objective_function)