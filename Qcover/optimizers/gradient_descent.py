# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Notice: This document has been modified on the basis of the original code

import numpy as np
from typing import Iterator, Optional, Union, Callable, Dict, Any
from functools import partial

CALLBACK = Callable[[int, np.ndarray, float, float], None]


class GradientDescent:

    def __init__(self,
            # p: int = 1,
            maxiter: int = 100,
            learning_rate: Union[float, Callable[[], Iterator]] = 0.0005,
            tol: float = 1e-7,
            callback: Optional[CALLBACK] = None,
            perturbation: Optional[float] = None,
            initial_point: Optional[np.ndarray] = None) -> None:

        self._p = None
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.perturbation = perturbation
        self.tol = tol
        self.callback = callback
        self._max_evals_grouped = 1
        self._initial_point = initial_point

    @property
    def settings(self) -> Dict[str, Any]:
        # if learning rate or perturbation are custom iterators expand them
        if callable(self.learning_rate):
            iterator = self.learning_rate()
            learning_rate = np.array([next(iterator) for _ in range(self.maxiter)])
        else:
            learning_rate = self.learning_rate

        return {
            "maxiter": self.maxiter,
            "tol": self.tol,
            "learning_rate": learning_rate,
            "perturbation": self.perturbation,
            "callback": self.callback,
        }

    @staticmethod
    def gradient_num_diff(x_center, f, epsilon, max_evals_grouped=1):
        """
        We compute the gradient with the numeric differentiation in the parallel way,
        around the point x_center.

        Args:
            x_center (ndarray): point around which we compute the gradient
            f (func): the function of which the gradient is to be computed.
            epsilon (float): the epsilon used in the numeric differentiation.
            max_evals_grouped (int): max evals grouped
        Returns:
            grad: the gradient computed

        """
        forig = f(*((x_center,)))
        grad = []
        ei = np.zeros((len(x_center),), float)
        todos = []
        for k in range(len(x_center)):
            ei[k] = 1.0
            d = epsilon * ei
            todos.append(x_center + d)
            ei[k] = 0.0

        counter = 0
        chunk = []
        chunks = []
        length = len(todos)
        # split all points to chunks, where each chunk has batch_size points
        for i in range(length):
            x = todos[i]
            chunk.append(x)
            counter += 1
            # the last one does not have to reach batch_size
            if counter == max_evals_grouped or i == length - 1:
                chunks.append(chunk)
                chunk = []
                counter = 0

        for chunk in chunks:  # eval the chunks in order
            parallel_parameters = np.concatenate(chunk)
            todos_results = f(parallel_parameters)  # eval the points in a chunk (order preserved)
            if isinstance(todos_results, float):
                grad.append((todos_results - forig) / epsilon)
            else:
                for todor in todos_results:
                    grad.append((todor - forig) / epsilon)

        return np.array(grad)

    def _minimize(self, loss, grad):   #, initial_point
        # set learning rate
        if isinstance(self.learning_rate, float):
            eta = constant(self.learning_rate)
        else:
            eta = self.learning_rate()

        if grad is None:
            eps = 0.01 if self.perturbation is None else self.perturbation
            grad = partial(
                GradientDescent.gradient_num_diff,
                f=loss,
                epsilon=eps,
                max_evals_grouped=self._max_evals_grouped,
            )

        # prepare some initials
        x = np.asarray(self._initial_point)
        nfevs = 0

        for _ in range(1, self.maxiter + 1):
            # compute update -- gradient evaluation counts as one function evaluation
            update = grad(x)
            nfevs += 1

            # compute next parameter value
            x_next = x - next(eta) * update

            # send information to callback
            stepsize = np.linalg.norm(update)
            if self.callback is not None:
                self.callback(nfevs, x_next, loss(x_next), stepsize)

            # update parameters
            x = x_next

            # check termination
            if stepsize < self.tol:
                break

        return x, loss(x), nfevs

    def optimize(self, objective_function, p, gradient_function=None):
        if self._initial_point is None:
            self._initial_point = np.array([np.random.random() for x in range(2 * p)])
        return self._minimize(objective_function, gradient_function)  #, initial_point


def constant(eta=0.01):
    """Yield a constant."""

    while True:
        yield eta
