# pylint: disable=invalid-name,wrong-import-position

"""Main QCover public functionality."""

import pkgutil
import sys
import warnings
import os

# Allow extending this namespace. Please note that currently this line needs
# to be placed *before* the wrapper imports or any non-import code AND *before*
# importing the package you want to allow extensions for (in this case `backends`).
__path__ = pkgutil.extend_path(__path__, __name__)
__version__ = '1.0.7'
__license__ = 'Apache-2.0 License'

from Qcover.core import *
from Qcover.applications import *
from Qcover.backends import *
from Qcover.optimizers import *


