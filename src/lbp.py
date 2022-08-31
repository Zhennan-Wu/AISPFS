##############################################################################################
# Bottom implementation based on the reference: https://github.com/krashkov/Belief-Propagation
##############################################################################################

import numpy as np
import time as timing
import random

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


import jax
# jax.config.update('jax_platform_name', 'cpu')
from pgmax import infer

from utils import generate_factor_dist, reduce_to_factor
from factor_graph import factor_graph, plot_factor_graph
from bp import loopy_belief_propagation


def lbp_fwd_main(parser, policy, connect_dist, horizon, init_state, iter_num=1, connect_debug = False):
    '''
    Forward loopy belief propagation.
    '''
    graph_debug = False
    lbp_lst = []
    s_vars = parser.get_state_vars()
    a_vars = parser.get_action_vars()
    valid_actions = parser.get_valid_action_lst()
    trans_prob = parser.get_pseudo_trans()
    state_dependency = parser.get_state_dependency()
    normal_factor = parser.get_normal_factor()
    reward_dist = parser.get_reward_table()
    atomic_action_lst = parser.get_atomic_action_lst()

    for action in valid_actions:
        # constructing factor graph
        init_factor_nodes, policy_factor_nodes, trans_factor_nodes, partial_reward_factor_nodes, cumu_reward_factor_nodes, connect_factor_nodes = reduce_to_factor(state_dependency, valid_actions, atomic_action_lst, reward_dist, connect_dist, trans_prob, policy, s_vars, a_vars, horizon, init_state, 'fw', normal_factor, connect_debug, action)

        factored_graphDBN = factor_graph()
        for time_step in range(0, horizon):
            time_stamp = str(time_step)
            next_time = time_step + 1
            next_stamp = str(next_time)

            for i, node in enumerate(policy_factor_nodes[time_step]):
                index = str(i)
                node_name = "policy" + time_stamp + '_' + index
                factored_graphDBN.add_factor_node(node_name, node)

            if (time_step == 0):
                for i, node in enumerate(init_factor_nodes[time_step]):
                    index = str(i)
                    node_name = 'init' + time_stamp + '_' + index
                    factored_graphDBN.add_factor_node(node_name, node)
            else:
                for i, node in enumerate(trans_factor_nodes[time_step]):
                    index = str(i)
                    node_name = "trans" + time_stamp + '_' + index
                    factored_graphDBN.add_factor_node(node_name, node)

            for i, node in enumerate(partial_reward_factor_nodes[next_time]):
                index = str(i)
                node_name = "partial reward" + next_stamp + '_' + index
                factored_graphDBN.add_factor_node(node_name, node)

            for i, node in enumerate(cumu_reward_factor_nodes[next_time]):
                index = str(i)
                node_name = "pseudo reward" + next_stamp + '_' + index
                factored_graphDBN.add_factor_node(node_name, node)

            for i, node in enumerate(connect_factor_nodes[next_time]):
                index = str(i)
                node_name = "cumu reward" + next_stamp + '_' + index
                factored_graphDBN.add_factor_node(node_name, node)
        if (graph_debug):
            plot_factor_graph(factored_graphDBN, "./newfactoredDBN.html")

        # Loopy Belief Propagation
        lbp = loopy_belief_propagation(factored_graphDBN)
        un_normal_prob = lbp.belief("c"+str(horizon), iter_num).get_distribution()
        prob = un_normal_prob/np.sum(un_normal_prob)
        lbp_lst.append(prob[1])

    best_prob = max(lbp_lst)
    candid_idx = []
    for idx, prob in enumerate(lbp_lst):
        if (prob == best_prob):
            candid_idx.append(idx)
    best_a_idx = random.choices(candid_idx)[0]

    return valid_actions[best_a_idx]    


def lbp_fwd_main_jax(parser, alg_struc, policy, horizon, init_state, iter_num):
    '''
    Forward loopy belief propagation.
    '''
    graph_debug = False
    lbp_lst = []
    s_vars = parser.get_state_vars()
    a_vars = parser.get_action_vars()
    valid_actions = parser.get_valid_action_lst()
    atomic_action_lst = parser.get_atomic_action_lst()

    fg = alg_struc[0]
    init_candids = alg_struc[1]
    policy_factors = alg_struc[2]
    initial_factors = alg_struc[3]

    # constructing factor graph
    init_f_dists = []
    for idx, s_var in enumerate(init_candids):
        s_grounding = float(init_state[s_var])
        init_f_dists.append(jax.numpy.array([1. - max(s_grounding - 0.0001, 0.0001), max(s_grounding - 0.0001, 0.0001)]))
    init_probs = np.stack(init_f_dists, axis=0)

    for action in valid_actions:
        policy_f_dists = []
        for time_step in range(0, horizon):
            time_stamp = str(time_step)
            policy_factor_vars = []
            factor_var = []
            factor_var.append("t" + time_stamp + "_atomic_action")
            policy_factor_vars.append(factor_var)
            for idx, factor_var in enumerate(policy_factor_vars):
                policy_f_dist = generate_factor_dist(time_step, s_vars, a_vars, valid_actions, atomic_action_lst, "policy", factor_var, policy, 'fw', action)
                policy_f_dists.append(policy_f_dist)
        policy_probs = np.stack(policy_f_dists, axis=0)

        found = False
        bp = infer.BP(fg.bp_state, temperature=1.0)
        bp_arrays = bp.init(
            log_potentials_updates={
                initial_factors: jax.numpy.log(init_probs),
                policy_factors: jax.numpy.log(policy_probs)
        }
        )
        bp_arrays = bp.run_bp(bp_arrays, num_iters=iter_num, damping=0)
        beliefs = bp.get_beliefs(bp_arrays)
        marginals = infer.get_marginals(beliefs)
        # potential bug to find the right variable
        for var_type in marginals.keys():
            if (var_type.shape[0] == horizon+1):
                prob = marginals[var_type][horizon, 1]
                lbp_lst.append(prob)
                found = True
        if (not found):
            raise Exception("Something is wrong")
    best_prob = max(lbp_lst)
    candid_idx = []
    for idx, prob in enumerate(lbp_lst):
        if (prob == best_prob):
            candid_idx.append(idx)
    best_a_idx = random.choices(candid_idx)[0]

    return valid_actions[best_a_idx]


def lbp_bwd_main(parser, alg_struc, init_state, iter_num):
    '''
    Backward loopy belief propagation.
    '''
    # constructing factor graph
    atomic_action_lst = parser.get_atomic_action_lst()

    fg = alg_struc[0]
    init_candids = alg_struc[1]
    initial_factors = alg_struc[3]

    init_f_dists = []
    for s_var in init_candids:
        s_grounding = float(init_state[s_var])
        init_f_dists.append(jax.numpy.array([1. - max(s_grounding - 0.0001, 0.0001), max(s_grounding - 0.0001, 0.0001)]))
    probs = np.stack(init_f_dists, axis=0)

    lbp_lst = []
    un_normal_prob = []
    bp = infer.BP(fg.bp_state, temperature=1.0)
    bp_arrays = bp.init(
        log_potentials_updates={
            initial_factors: jax.numpy.log(probs)
        }
    )

    bp_arrays = bp.run_bp(bp_arrays, num_iters=iter_num, damping=0)
    beliefs = bp.get_beliefs(bp_arrays)
    marginals = infer.get_marginals(beliefs)
    for var_type in marginals.keys():
        if (var_type.variables[0][1] == len(atomic_action_lst)):
            un_normal_prob = marginals[var_type][0]
    # Loopy Belief Propagation

    un_normal_prob = np.array(un_normal_prob)
    prob = un_normal_prob/np.sum(un_normal_prob)
    lbp_lst.append(prob.tolist())

    return lbp_lst
