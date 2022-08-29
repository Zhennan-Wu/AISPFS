# Code structure
## General
- `main.py`: Reading parameters from configuration files and conducting the experiments.
- `spudd_parser.py`: Reading and transform domain instance information from SPUDD file for algorithmic usage.
- `utils.py`: Implementation of policy update procedure, output to file function, and etc.. 
- `simu.py`: Parallel simulation of the algorithms.
## Variational Inference (MFVI, CSVI)
- `vi.py`: Implementation of MFVI forward and backward methods.
- `vi_noS`: Implementation of a limited version of MFVI forward and backward methods that do not update state approximate distribution.
- `exp_vi`: Implementation of MFVI forward and backward methods with exponentiated reward.
- `act_only.py`: Implementation of CSVI forward and backward methods.
- `exp_act_only.py`: Implementation of CSVI forward and backward methods with exponentiated reward.
- `variational.py`: Low level implementation of MFVI approximate distribution update procedure, CFVI sampling estimation procedure and ELBO calculation.
- `demo_vi.py`: Variational inference methods (MFVI, CSVI) for handcrafted demo domain, the algothms are the same but with a more detailed output (approximate distribution printout, ELBO decomposition plot, etc.). 
## Loopy Belief Propagation
- `lbp.py`: Implementation of loopy BP forward and backward methods.
- `bp.py`: Implementation of loopy BP message passing mechanism.
- `factor_graph.py`: Bottom implementation of loopy BP graphical structure.
- `utils.py`: Implementation of the creation of the corresponding graphical sturcture from instance information.
## Visualization
- `temp_elbo_plot.py`: Visualization of ELBO decomposition of different class of random variables.
- `visual.py`: Visualization of algorithm performance in a domain by domain manner.
- `summary.py`: Visualization of algorithm performance across domains in one plot.
- `demo_visual.py`: Visualization of algorithm performance in an instance by instance manner. It is created separately because we conducted different algorithms on demo for analysis.

