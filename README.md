# Approximate Inference for Stochastic Planning in Factored Space

## Usage
- Download the repository and unzip the planning domain file "spudd_sperseus.zip" under the current directory.
- Install the required packages `pip install -r requirements`.
- To run the experiments run `python3 main.py -i <instance full name> -a <algorithms to run> `. For example, `python3 main.py -i sysadmin_inst_mdp__1.spudd elevators_inst_mdp__5.spudd sysadmin_inst_mdp__2.spudd -a mfvi_bwd fwd_lbp random`. If given no parameter, the program will run all 60 instances from 6 domains on all algorithms (mfvi_bwd, mfvi_fwd, mfvi_bwd_noS, bwd_lbp, fwd_lbp, csvi_bwd, csvi_fwd, exp_mfvi_bwd, exp_csvi_bwd). 
  - For running the demo problem, use (demo_mfvi_bwd, demo_mfvi_bwd_med, demo_mfvi_bwd_noS, demo_mfvi_bwd_main_all, demo_csvi_bwd, random) for algorithm parameters to get the detailed plotting and print out. The demo instance names include (demo_inst_mdp__1, demo_inst_mdp__2, demo_inst_mdp__3). To regenerate the ELBO plotting in other problem domains, please run the corresponding demo version of the algorithm on the problem instance.
- To visulize the results
  -   `python3 summary.py -a <algorithm to visualize>` will generate the summary comparison plots among all the domains, to facilitate plotting, please put 'random' as the last algorithm.
  - `python3 visual.py -d <domain> -a <algorithm to visualize>` will generate the algorithm comparison plots of the domain. If given no parameter, it will generate plots on all 6 domains of algorithms mfvi_bwd, csvi_bwd, fwd_lbp, sogbofa, random.
- Please refer to `python3 main.py -h`, `python3 summary.py -h`, `python3 visual.py -h` for more information. 
- All the algorithmic and expermental parameters are stored in `./conf/alg_param.json` and should be modified to align with running time and computational resourse contrains.
- The results in the paper are stored in `./results_of_paper.zip`. To visulize based on those results, please unzip it into the folder `results`.
- To compare with sogbofa, please unzip `sogbofa.zip` and put the results of sogbofa inside `./results/`.
- To regenerate the results of sogbofa, see the **README** file inside the **SOGBOFA** directory.
- Notice that the code does not erase the previous running results in order to faciliate parallel running and data collection. Thus running the same algorithm on the same instance multiple times will accumulate the results in the same file, which might cause problems during visualization if the algorithms under comparison have a different number of results. To avoid it, please clean the `results` folder every time before running. 
- To visualize CSVI w.r.t iteration number, one convenient way is to use the original results in `results_of_paper.zip`. You could also generate new results by tuning parameters in `./conf/alg_param.json`, but need to put the results in `./results/iter_comp/iter1` and `./results/iter_comp/iter100` respectively to make the visulization work.
- Before running, for things to work normally, the directory structure should look like (The `logs` directory is optional and will be automatically generated and the `results` directory is optional if not using the pre-generated results.)
- If you run into memory issue in running backward loopy Belief Propagation, that is because jax by default preallocates the memory. Please refer to [Jax Doc](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html) for corresponding modification or switch to CPU instead by setting `jax.config.update('jax_platform_name', 'cpu')` in `utils.py`.

```
project
|
│   README.md
│   requirements.txt
|   results_of_paper.zip
|   sogbofa.zip
|   spudd_sperseus.zip
|
└───conf
│   │   default.conf
│   │   alg_param.json
|   |   ...
└───results
│   └───alg1
│       │   <instance>_<alg1>_rlt.json
│       │   ...
│   └───alg2
|       │   ...
|   ...
└───spudd_sperseus
│   │   <instance>.spudd
│   │   ...
└───src
│   │   main.py
│   │   ...
└───logs
│   └───alg1
|   |   |   ...
```
## Acknowledgement
- The code of second forward loopy Belief Propagation is based on and modified from [Aleksei Krasikov's github repository](https://github.com/krashkov/Belief-Propagation).
- The code of backward loopy Belief Propagation is based on [PGMax](https://github.com/vicariousinc/PGMax).
- The code of SOGBOFA is based on and modified from [Hao Cui's github repository](https://github.com/hcui01/SOGBOFA).

