import sys
import numpy as np
from typing import Iterator, Optional, Union, Callable, Dict, Any
from Qcover.exceptions import ArrayShapeError

class GradientDescent:

    def __init__(self,
                 options: dict = None,  # {'maxiter':300, 'learning_rate':0.0005, 'tol':1e-6, 'perturbation':0.1},
                 initial_point: Optional[np.ndarray] = None) -> None:

        super().__init__()

        self._p = None
        self._options = options
        self._initial_point = initial_point

    def _minimize(self, loss):
        if 'learning_rate' in self._options:
            lr = self._options["learning_rate"]
        else:
            lr = 0.01

        if 'tol' in self._options:
            tol = self._options['tol']
        else:
            tol = 1e-6

        if 'maxiter' in self._options:
            maxiter = self._options['maxiter']
        else:
            maxiter = 500

        eps = 0.01 if "perturbation" not in self._options else self._options['perturbation']

        # prepare some initials
        x = np.asarray(self._initial_point)
        nfevs = 0

        for _ in range(1, maxiter + 1):
            delta = np.random.random() - 0.5
            # delta = np.random.randint(0, 1, size=len(self._initial_point)) - 0.5
            diff = loss(x + delta) - loss(x - delta)  #
            grad = 0.5 * diff / delta  #

            x_next = x - lr * grad
            stepsize = np.linalg.norm(grad)  #the distance between grad to zero
            x = x_next
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



