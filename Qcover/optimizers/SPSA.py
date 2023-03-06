import sys
from typing import Optional
import logging
import numpy as np
from scipy import optimize as opt
from Qcover.optimizers import Optimizer
from Qcover.exceptions import ArrayShapeError
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class SPSA(Optimizer):
    # pylint: disable=unused-argument
    def __init__(self,
                 options: dict = None,  # {'maxiter':300, 'disp':True, 'rhobeg': 1.0, 'tol':1e-6},
                 initial_point: Optional[np.ndarray] = None) -> None:
        """
        Args:
            options: some optional setting parameters such as:
                maxiter: Maximum number of function evaluations.
                disp: Set to True to print convergence messages.
                rhobeg: Reasonable initial changes to the variables.
                tol: Final accuracy in the optimization (not precisely guaranteed).
                     This is a lower bound on the size of the trust region.
        """
        super().__init__()
        self._p = None
        self._options = options
        self._initial_point = initial_point

    def _minimize(self, loss):
        if 'A' in self._options:
            A = self._options["A"]
        else:
            A = 20

        if 'R' in self._options:
            R = self._options['R']
        else:
            R = 0.2

        if 'a0' in self._options:
            a0 = self._options["a0"]
        else:
            a0 = 1.3

        if 'c0' in self._options:
            c0 = self._options['c0']
        else:
            c0 = 0.2

        if 'tol' in self._options:
            tol = self._options['tol']
        else:
            tol = 1e-6

        if 'maxiter' in self._options:
            maxiter = self._options['maxiter']
        else:
            maxiter = 500

        # a0 = 1.2
        # c0 = 1.0
        # prepare some initials
        x = np.asarray(self._initial_point)
        nfevs = 0

        for i in range(1, maxiter + 1):
            a = a0 / (i + 1) ** A
            c = c0 / (i + 1) ** R
            # delta = np.random.random() - 0.5
            delta = (2 * np.random.randint(0, 2, size=len(self._initial_point)) - 1) * c
            diff = loss(x + delta) - loss(x - delta)  #
            grad = 0.5 * diff / delta  #

            x_next = x - a * grad
            x = x_next
            stepsize = np.linalg.norm(grad)  # the distance between grad to zero
            nfevs += 1
            # check termination
            if stepsize < tol:
                break

        return x, loss(x), nfevs

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