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
    from Qcover.utils import generate_graph_data, generate_weighted_graph
    import networkx as nx
    
    node_num, edge_num = 6, 9
    p = 1
    nodes, edges = generate_graph_data(node_num, edge_num)
    g = generate_weighted_graph(nodes, edges)
   
    # If you want to customize the Ising weight graph model, you can use the following code
    # g = nx.Graph()
    # nodes = [(0, 3), (1, 2), (2, 1), (3, 1)]
    # edges = [(0, 1, 1), (0, 2, 1), (3, 1, 2), (2, 3, 3)]
    # for nd in nodes:
    #    u, w = nd[0], nd[1]
    #    g.add_node(int(u), weight=int(w))
    # for ed in edges:
    #     u, v, w = ed[0], ed[1], ed[2]
    #     g.add_edge(int(u), int(v), weight=int(w))

    qulacs_bc = CircuitByQulacs()
    optc = COBYLA(options={'tol': 1e-3, 'disp': True})
    qc = Qcover(g, p=p, optimizer=optc, backend=qulacs_bc)
    res = qc.run()
    print("the result of problem is:\n", res)
    qc.backend.optimization_visualization()
    ```
2. Solving specific binary combinatorial optimization problems, Calculating the expectation value of the Hamiltonian of the circuit which corresponding to the problem.
for example, if you want to using Qcover to solve a max-cut problem, just coding below:
    ```python
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
    counts = qc.backend.get_result_counts(res['Optimal parameter value'])
    qc.backend.sampling_visualization(counts)
    ```
   
3. If you want to solve combinatorial optimization problems with real quantum computers on
 quafu quantum computing cloud platform, you can refer to the following example code
    ```python
    from Qcover.core import Qcover
    from Qcover.backends import CircuitByQulacs
    from Qcover.optimizers import COBYLA
    from Qcover.compiler import CompilerForQAOA
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Qcover supports real quantum computers to solve combinatorial optimization problems.
    # You only need to transform the combinatorial optimization problem into a weight graph,
    # and you can use the quafu quantum computing cloud platform  (http://quafu.baqis.ac.cn/)
    # to solve the corresponding problem. The following is an example of a max-cut problem.
    
    # The weight graph corresponding to the combinatorial optimization problem and transformed it to networkx format.
    nodes = [(0, 1), (1, 3), (2, 2), (3, 1), (4, 0), (5, 3)]
    edges = [(0, 1, -1), (1, 2, -4), (2, 3, 2), (3, 4, -2), (4, 5, -1), (1, 3, 0), (2, 4, 3)]
    graph = nx.Graph()
    for nd in nodes:
        u, w = nd[0], nd[1]
        graph.add_node(int(u), weight=int(w))
    for ed in edges:
        u, v, w = ed[0], ed[1], ed[2]
        graph.add_edge(int(u), int(v), weight=int(w))
    
    # draw weighted graph to be calculated
    new_labels = dict(map(lambda x: ((x[0], x[1]), str(x[2]['weight'])), graph.edges(data=True)))
    pos = nx.spring_layout(graph)
    # pos = nx.circular_layout(g)
    nx.draw_networkx(graph, pos=pos, node_size=400, font_size=13, node_color='y')
    # nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=new_labels, font_size=15)
    nx.draw_networkx_edges(graph, pos, width=2, edge_color='g', arrows=False)
    plt.show()
    
    # Using Qcover to calculate the optimal parameters of QAOA circuit.
    p = 1
    bc = CircuitByQulacs()
    optc = COBYLA(options={'tol': 1e-3, 'disp': True})
    qc = Qcover(graph, p=p, optimizer=optc, backend=bc)
    res = qc.run()
    optimal_params = res['Optimal parameter value']
    
    # Compile and send the QAOA circuit to the quafu cloud.
    # Token parameter should be set according to your own account
    # For more introduction see https://github.com/ScQ-Cloud/pyquafu
    token = "E-SowFQdKJ427YhZDGdxoNmOk2SB02xpgODiz_4WtAS.9dDOwUTNxgjN2EjOiAHelJCLzITM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye"
    cloud_backend = 'ScQ-P20'
    qcover_compiler = CompilerForQAOA(graph, p=p, optimal_params=optimal_params, apitoken=token, cloud_backend=cloud_backend)
    task_id = qcover_compiler.send(wait=True, shots=5000, task_name='MaxCut')
    # If you choose wait=Ture, you have to wait for the result to return.
    # If you choose wait=False, you can execute the following command to query the result status at any time,
    # and the result will be returned when the task is completed.
    quafu_solver = qcover_compiler.task_status_query(task_id)
    if quafu_solver:
        counts_energy = qcover_compiler.results_processing(quafu_solver)
        qcover_compiler.visualization(counts_energy, problem='MaxCut', solutions=3)
    ```

The results obtained by running this example code are shown in the following two figures

<div align="center">
  <img src=./tests/test_compiler_graph.png width="300"/><img src=./tests/test_compiler_res.png width="300"/>
</div>

# How to contribute
For information on how to contribute, please send an e-mail to members of developer of this project.

# Please cite
When using Qcover for research projects, please cite

- Wei-Feng Zhuang, Ya-Nan Pu, Hong-Ze Xu, Xudan Chai, Yanwu Gu, Yunheng Ma, Shahid Qamar, 
Chen Qian, Peng Qian, Xiao Xiao, Meng-Jun Hu, and Done E. Liu, "Efficient Classical
Computation of Quantum Mean Value for Shallow QAOA Circuits", arXiv:2112.11151 (2021).


- BAQIS Quafu Group, "Quafu-Qcover: Explore Combinatorial Optimization Problems on Cloud-based Quantum Computers", arXiv:2305.17979 (2023).


# Authors
The first release of Qcover was developed by the quantum operating system team in Beijing Academy of Quantum Information Sciences.

Qcover is constantly growing and many other people have already contributed to it in the meantime.

# License
Qcover is released under the Apache 2 license.
