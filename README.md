# Qcover - A QAOA based combinational optimization solver
Qcover is an open source effort to help exploring combinatorial optimiaztion problems in Noisy Intermediate-scale Quantum(NISQ) processor. It is developed by the quantum operating system team in Beijing Academy of Quantum Information Sciences (BAQIS). Qcover supports fast output of optimal parameters in shallow QAOA circuits. It can be used as a powerful tool to assist NISQ processor to demonstrate application-level quantum advantages. 


# Getting started
To start using Qcover, simply run
```
python -m pip install --user qcover
```
More example codes and tutorials can be found in the tests folder here on GitHub.

Also, make sure to check out the detailed code documentation.

# Examples
```python
frome core import QCover
from backends import CircuitByQulacs
from optimizers import COBYLA, Fourier
from networkx import Graph

nodes, edges = QCover.generate_graph_data(6, 9)
g = QCover.generate_weighted_graph(nodes, edges)
qulacs_bc = CircuitByQulacs()
qc = QCover(g, p, optimizer=optc, backend=qulacs_bc)
res = qc.run(is_parallel=True)
```


# How to contribute
For information on how to contribute, please send an e-mail to members of developer of this project.

# Please cite
When using Qcover for research projects, please cite

- 


# Authors
The first release of Qcover (v1.0.0) was developed by the quantum operating system team in Beijing Academy of Quantum Information Sciences.

Qcover is constantly growing and many other people have already contributed to it in the meantime.

# License
Qcover is released under the Apache 2 license.
