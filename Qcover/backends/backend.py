"""Backend interface"""

from enum import IntEnum
import logging
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name

class Backend(ABC):
    """Base class for backend."""

    # @abstractmethod
    # def __init__(self, *args):
    #     pass

    @abstractmethod
    def __init__(self,
                 p: int = 1,
                 nodes_weight: list = None,
                 edges_weight: list = None,
                 is_parallel: bool = None) -> None:
        self._p = p
        self._nodes_weight = nodes_weight
        self._edges_weight = edges_weight
        self._is_parallel = False if is_parallel is None else is_parallel

        self._element_to_graph = None
        self._pargs = None
        self._element_expectation = dict()

    # @abstractmethod
    # def get_operator(self, *args):
    #     pass

    @abstractmethod
    def get_expectation(self, *args):
        pass

    @abstractmethod
    def expectation_calculation(self):
        pass

    @abstractmethod
    def get_result_counts(self, params):
        pass

    @abstractmethod
    def optimization_visualization(self):
        pass

    @abstractmethod
    def sampling_visualization(self, counts):
        pass