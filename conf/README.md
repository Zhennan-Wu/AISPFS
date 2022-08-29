`alg_param.json` and `default.conf` could be adjust for convenience.
# Parameters in alg_param.json
- `"epids"`:  Number of simulation to run on one instance, the paper setting is **12**.
- `"em_steps"`: Number of policy update in the variational inference forward method, the paper setting is **3**.
- `"look_ahead_depths"`: Search depth of all the algorithms, the paper setting is **9**.
- `"rwd_dist_sample_size"`:  Sample size of reward distribution estimation, the paper setting is **80**.
- `"act_seq_sample_size"`: Sample size of action sequence in CSVI, the paper setting is **20**.
- `"full_traj_sample_size"`: Sample size of the full trajectory in CSVI, the paper setting is **50**.
- `"fwd_lbp_iter_num"`: Forward loopy BP iteration number, the paper setting is **1**.
- `"bwd_lbp_iter_num"`: Backward loopy BP iteration number, the paper setting is **50**.
- `"mfvi_max_iter"`: MFVI maximum number of approximate distribution update, the paper setting is **100**.
- `"csvi_max_iter"`: CSVI maximum number of approximate distribution update, the paper setting is **1**. During experiment w.r.t the iteration number it is set to be **1** and **100** respectively.
- `"vi_conv_threshold"`: Convergence threshold of variational inference methods, the paper setting is **0.1**.
