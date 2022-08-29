# OVERVIEW

SOGBOFA is a state-of-the-art stochastic online planning algorithm that works well in MDPs that have large state and action spaces. It 
uses the idea of "Aggregate Simulation" to build a computation graph which symbolically approximates the Q value of actions given the 
current state, and searches the action space very efficiently using gradient updates. It supports various action constraints defined in 
first-order predicates. It directly works with MDP domains defined in RDDL, and is implemented based on the original code of the RDDL 
simulator (https://github.com/ssanner/rddlsim). 

This is the modification version of [the original SOGBOFA code](https://github.com/hcui01/SOGBOFA) with the addition of fixed search depth and fixed number of gradient updates.

# Usage

- Prerequisites: Java SE 1.8 or higher
- Compile
  - In the SOGBOFA/ directory, type command `./compile`
- Run the server
  - One can either run a server from the RDDLSim Project, or run the Server class compiled from our source code with the following command
  `./run_server rddlfilename-or-dir portnumber num-rounds random-seed timeout` 
  - For example: 
  `./run_server Domains 2323 12 1 50000`
  - Note that the time out is the total time (in seconds) allowed to any client connects to it.
  - In the original SOGBOFA code, the algorithm will adaptively choose search depth and gradient update steps to take full advantage of the remaining time. However in our modified version if it finishes the search earlier it will proceed and abort the remaining time. `50000` is tested sufficient for SOGBOFA to finish the search of depth 9 and 500 gradient update among all the domains. To facilitate the result format transferring please keep it that way.
- Run SOGBOFA
  - Type the following command
  `./run_sogbofa instance-name portnumber fixed-search-depth-flag(true or false) search-depth gradient-update-number`
  - Example:
  `./run_sogbofa traffic_inst_mdp__10 2323 true 9 500`
  (the setting used in the paper) Notice that when the fixed-search-depth-flag is set to be false, the search-depth and gradient-update-number parameters will not affect the algorithm, but they need to be set to arbitrary numbers to enable the algorithm run normally.
- Results
  After finishing the requested number of runs, 6 files will be recorded in the bin/ folder, with the following names
  - L_C_SOGBOFA_timeout_instance-name_Score
  - L_C_SOGBOFA_timeout_instance-name_depthCounter
  - L_C_SOGBOFA_timeout_instance-name_sizeCounter
  - L_C_SOGBOFA_timeout_instance-name_rrCounter
  - L_C_SOGBOFA_timeout_instance-name_seenCounter
  - L_C_SOGBOFA_timeout_instance-name_updatesCounter
  which record respectively the average score of the runs, the average depth of the search, the average size of the computation graph, 
  the average number of radnom restarts, the average number of action evaluated, the average number of gradient updates.
  The Score file has three numbers, respectively the instance index, the average cumulative reward, and the standard deviation. The other
  files each has only one number inside.

- To run the code with no lifting or conformant, as a sanity check for our implementation of the forward loopy Belief Propagation algorithm, you can type
   `./run_sogbofa_original instance-name portnumber fixed-search-depth-flag(true or false) search-depth gradient-update-number`.
In this way, the names of the results files should begin with "SOGBOFA_Original" instead of "L_C_SOGBOFA".

- To prepare for the comparison among algorithms, one need to run the python script `python3 rlt_transfer.py sogbofa` or `python3 rlt_transfer.py sogorigin` to transform the results of SOGBOFA or SOGBOFA without gradient update into the same formats as other algorithm results and put it in the corresponding directory.
