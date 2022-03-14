<div align="center">
  <img src=./resources/Qcover_label_readme.png>
</div>
Qcover is an open source effort to help exploring combinatorial optimization problems in Noisy Intermediate-scale Quantum(NISQ) processor. It is developed by the quantum operating system team in Beijing Academy of Quantum Information Sciences (BAQIS). Qcover supports fast output of optimal parameters in shallow QAOA circuits. It can be used as a powerful tool to assist NISQ processor to demonstrate application-level quantum advantages. 

# Getting started
Use the following command to complete the installation of Qcover
```git
pip install Qcover
```
or

```git
git clone https://github.com/BAQIS-Quantum/Qcover
pip install -r requirements.yml
python setup.py install
```
More example codes and tutorials can be found in the tests folder here on GitHub.

# Examples
1. Using algorithm core module to generate the ising random weighted graph and calculate it's Hamiltonian expectation
    ```python
    from Qcover.core import Qcover
    from Qcover.backends import CircuitByQulacs
    from Qcover.optimizers import COBYLA
    
    node_num, edge_num = 6, 9
    p = 1
    nodes, edges = Qcover.generate_graph_data(node_num, edge_num)
    g = Qcover.generate_weighted_graph(nodes, edges)
    qulacs_bc = CircuitByQulacs()
    optc = COBYLA(options={'tol': 1e-3, 'disp': True})
    qc = Qcover(g, p=p, optimizer=optc, backend=qulacs_bc)
    res = qc.run()
    print("the result of problem is:\n", res)
    qc.backend.visualization()
    ```
2. Solving specific binary combinatorial optimization problems, Calculating the expectation value of the Hamiltonian of the circuit which corresponding to the problem.
for example, if you want to using Qcover to solve a max-cut problem, just coding below:
    ```python
    import numpy as np
    from Qcover.core import Qcover
    from Qcover.backends import CircuitByQiskit
    from Qcover.optimizers import COBYLA
    from Qcover.applications.max_cut import MaxCut
    node_num, degree = 6, 3
    p = 1
    mxt = MaxCut(node_num=node_num, node_degree=degree)
    ising_g, shift = mxt.run()
    qiskit_bc = CircuitByQiskit(expectation_calc_method="statevector")
    optc = COBYLA(options={'tol': 1e-3, 'disp': True})
    qc = Qcover(ising_g, p=p, optimizer=optc, backend=qiskit_bc)
    res = qc.run()
    print("the result of problem is:\n", res)
    qc.backend.visualization()
    ```
3. If you want to customize the Ising weight graph model and calculate the ground
state expectation with Qcover, you can use the following code
    ```python
    import numpy as np
    import networkx as nx
    from Qcover.core import Qcover
    from Qcover.backends import CircuitByTensor
    from Qcover.optimizers import COBYLA

    ising_g = nx.Graph()
    nodes = [(0, 3), (1, 2), (2, 1), (3, 1)]
    edges = [(0, 1, 1), (0, 2, 1), (3, 1, 2), (2, 3, 3)]
    for nd in nodes:
       u, w = nd[0], nd[1]
       ising_g.add_node(int(u), weight=int(w))
    for ed in edges:
        u, v, w = ed[0], ed[1], ed[2]
    ising_g.add_edge(int(u), int(v), weight=int(w))

    p = 2
    optc = COBYLA(options={'tol': 1e-3, 'disp': True})
    ts_bc = CircuitByTensor()
    qc = Qcover(ising_g, p=p, optimizer=optc, backend=ts_bc)
    res = qc.run()
    print("the result of problem is:\n", res)
    qc.backend.visualization()
    ```

# How to contribute
For information on how to contribute, please send an e-mail to members of developer of this project.

# Please cite
When using Qcover for research projects, please cite

- Wei-Feng Zhuang, Ya-Nan Pu, Hong-Ze Xu, Xudan Chai, Yanwu Gu, Yunheng Ma, Shahid Qamar, 
Chen Qian, Peng Qian, Xiao Xiao, Meng-Jun Hu, and Done E. Liu, "Efficient Classical
Computation of Quantum Mean Value for Shallow QAOA Circuits", arXiv:2112.11151 (2021). 


# Authors
The first release of Qcover was developed by the quantum operating system team in Beijing Academy of Quantum Information Sciences.

Qcover is constantly growing and many other people have already contributed to it in the meantime.

# License
Qcover is released under the Apache 2 license.
