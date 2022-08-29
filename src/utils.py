import random
import numpy as np
import math
import itertools

import jax
# jax.config.update('jax_platform_name', 'cpu')

from pgmax import fgraph, vgroup, fgroup
from pgmax import factor as F
from factor_graph import factor



def record(x, path=None, flush=True):
    '''
    Print to file or stdout
    '''
    if path is None:
        print(x)
    else:
        print(x, file=open(path, 'a'), flush=flush)


def generate_connect_distribution(horizon):
    '''
    Generate probability table for auxiliary nodes connecting consecutive reward, represented as python dictionary.
    '''
    connect_dist = {}
    for time in range(0, horizon+1):
        if (time == 0):
            connect_dist[time] = {"00":0.9999, "01": 0.9999, "10": 0.9999, "11": 0.9999}
        elif (time == 1):
            connect_dist[time] = {"00":0.0001, "01": 0.9999, "10": 0.0001, "11": 0.9999}
        else:
            connect_dist[time] = {"00":0.0001, "01": 1.0/time, "10": 1.0-1.0/time, "11": 0.9999}
    return connect_dist


def generate_same_time_joint_cumu(rwd_case_num):
    '''
    Generate factored reward collecting structure distribution of the same time step.
    '''
    joint_cumu = []
    cumu_group = {}
    for idx in range(rwd_case_num):
        if (idx == rwd_case_num - 1):
            case = {'row_var': ['pc'+str(idx), 'pr'+str(idx+1), 'r']}
        else:
            case = {'row_var': ['pc'+str(idx), 'pr'+str(idx+1), 'pc'+str(idx+1)]}
        rwd_enum = list(itertools.product(['0', '1'], repeat=3))
        case['row_val'] = [''.join(g) for g in rwd_enum]
        if (idx == 0):
            case['table'] = np.array([0.9999, 0.0001, 0.0001, 0.9999, 0.9999,  0.0001, 0.0001, 0.9999])
        else:
            case['table'] = np.array([0.9999, 0.00001, 1. - 1./(idx+1), 1./(idx+1), 1./(idx+1), 1. - 1./(idx+1), 0.0001, 0.9999])
        joint_cumu.append(case)

        if ('pr'+str(idx+1) not in cumu_group.keys()):
            cumu_group['pr'+str(idx+1)] = [idx]
        else:
            cumu_group['pr'+str(idx+1)].append(idx)
        if (idx == rwd_case_num - 1):
            if ('r' not in cumu_group.keys()):
                cumu_group['r'] = [idx]
            else:
                cumu_group['r'].append(idx)
        else:
            if ('pc'+str(idx+1) not in cumu_group.keys()):
                cumu_group['pc'+str(idx+1)] = [idx, idx+1]
            else:
                cumu_group['pc'+str(idx+1)].append(idx)
                cumu_group['pc'+str(idx+1)].append(idx+1)

    return joint_cumu, cumu_group


def generate_cross_time_joint_cumu(horizon):
    '''
    Generate cumulative reward collecting structure distribution of the different time step.
    '''
    joint_cumu = []
    cumu_group = {}
    for idx in range(horizon):
        case = {'row_var': ['c'+str(idx), 'r'+str(idx+1), 'c'+str(idx+1)]}
        cumu_enum = list(itertools.product(['0', '1'], repeat=3))
        case['row_val'] = [''.join(g) for g in cumu_enum]
        # if (idx == 0):
        #     case['table'] = np.array([0.0001, 0.9999, 0.0001, 0.9999, 0.0001,  0.9999, 0.0001, 0.9999])
        if (idx == 0):
            case['table'] = np.array([0.9999, 0.0001, 0.0001, 0.9999, 0.9999,  0.0001, 0.0001, 0.9999])
        else:
            case['table'] = np.array([0.9999, 0.0001, 1. - 1./(idx+1), 1./(idx+1), 1./(idx+1), 1. - 1./(idx+1), 0.0001, 0.9999])
        joint_cumu.append(case)

        if ('c'+str(idx+1) not in cumu_group.keys()):
            cumu_group['c'+str(idx+1)] = [idx, idx+1]
        else:
            cumu_group['c'+str(idx+1)].append(idx)
            cumu_group['c'+str(idx+1)].append(idx+1)

        if ('r'+str(idx+1) not in cumu_group.keys()):
            cumu_group['r'+str(idx+1)] = [idx]
        else:
            cumu_group['r'+str(idx+1)].append(idx)

    return joint_cumu, cumu_group


def generate_vec_connect_distribution(horizon):
    '''
    Generate probability table for auxiliary nodes connecting consecutive reward, represented as python dictionary.
    '''
    space_enum = list(itertools.product(['0', '1'], repeat=3))
    space = [''.join(g) for g in space_enum]
    connect_dist = {}
    for time in range(0, horizon+1):
        if (time == 0):
            connect_dist[time] = np.array([[0.0001, 0.0001, 0.0001, 0.0001], [0.9999, 0.9999, 0.9999, 0.9999]])
        elif (time == 1):
            connect_dist[time] = np.array([[0.9999, 0.0001, 0.9999, 0.0001], [0.0001, 0.9999, 0.0001, 0.9999]])
        else:
            connect_dist[time] = np.array([[0.9999, 1. - 1.0/time, 1.0/time, 0.0001], [0.0001, 1.0/time, 1.0-1.0/time, 0.9999]])
    return connect_dist


def get_partial_reward(case, vars, state, normal_factor):
    '''
    Read reward from factored tree for belief propagtion
    '''
    r_min = normal_factor[0]
    r_max = normal_factor[1]
    epsilon = normal_factor[2]
    state_dict = {}
    for idx, svar in enumerate(vars):
        if (state[idx] == '1'):
            state_dict[svar] = 'true'
        else:
            state_dict[svar] = 'false'
    parent = case
    finished_search = False
    updated = False
    maxiter = len(state)
    count = 0
    rwd = 0.
    while (not finished_search and count <= maxiter):
        for key_p in parent.keys():
            for key_c in parent[key_p].keys():
                if (key_c == state_dict[key_p]):
                    parent = parent[key_p][key_c]
                    count += 1
                    updated = True
                    if (not type(parent) is dict):
                        rwd = float(parent)
                        finished_search = True
                    break
            if (updated):
                # reset updated for next round
                updated = False

            else:
                # enumerate all the keys in parent and did not find result, thus could terminate
                finished_search = True
    if (count > maxiter):
        raise Exception("reward tree is deeper than the number of state variables")

    return min((rwd - (r_min - epsilon))/(r_max - (r_min - epsilon)), 0.9999)


def get_exp_partial_reward(case, vars, state, normal_factor):
    '''
    Read exponentiated reward from factored tree for belief propagtion
    '''    
    r_min = normal_factor[0]
    r_max = normal_factor[1]
    epsilon = normal_factor[2]
    state_dict = {}
    for idx, svar in enumerate(vars):
        if (state[idx] == '1'):
            state_dict[svar] = 'true'
        else:
            state_dict[svar] = 'false'
    parent = case
    finished_search = False
    updated = False
    maxiter = len(state)
    count = 0
    rwd = 0.
    while (not finished_search and count <= maxiter):
        for key_p in parent.keys():
            for key_c in parent[key_p].keys():
                if (key_c == state_dict[key_p]):
                    parent = parent[key_p][key_c]
                    count += 1
                    updated = True
                    if (not type(parent) is dict):
                        rwd = float(parent)
                        finished_search = True
                    break
            if (updated):
                # reset updated for next round
                updated = False

            else:
                # enumerate all the keys in parent and did not find result, thus could terminate
                finished_search = True
    if (count > maxiter):
        raise Exception("reward tree is deeper than the number of state variables")

    return min(math.exp(rwd)/math.exp(r_max), 0.9999)


def generate_partial_reward_distribution(var_num, case, vars, normal_factor, state = ''):
    '''
    Recursively generate factored reward probability table on states for belief propagtion.
    '''
    if (var_num == 0):
        rwd = get_partial_reward(case, vars, state, normal_factor)
        # handle auxiliary binary univariate atomic_action variable
        prob = [1. - rwd, rwd]
        return prob
    else:
        return [generate_partial_reward_distribution(var_num-1, case, vars, normal_factor, state+'0'), generate_partial_reward_distribution(var_num-1, case, vars, normal_factor, state+'1')]


def generate_exp_partial_reward_distribution(var_num, case, vars, normal_factor, mes_type, state = ''):
    '''
    Recursively generate factored exponentiated reward probability table on states for belief propagtion.
    '''    
    if (var_num == 0):
        rwd = get_exp_partial_reward(case, vars, state, normal_factor)
        # handle auxiliary binary univariate atomic_action variable
        if (mes_type == 'bw'):
            prob = [0., rwd]
        else:
            prob = [1. - rwd, rwd]
        return prob
    else:
        return [generate_exp_partial_reward_distribution(var_num-1, case, vars, normal_factor, mes_type, state+'0'), generate_exp_partial_reward_distribution(var_num-1, case, vars, normal_factor, mes_type, state+'1')]


def generate_partial_a_reward_distribution(reward_table, normal_factor, mes_type, a_mask):
    '''
    Recursively generate factored reward probability table on actions for belief propagtion.
    '''
    r_min = normal_factor[0]
    r_max = normal_factor[1]
    epsilon = normal_factor[2]
    prob = []

    for atomic_action in reward_table.keys():
        raw = reward_table[atomic_action]['extra']
        rwd = min((raw - (r_min - epsilon))/(r_max - (r_min - epsilon)), 0.9999)
        prob.append([1. - rwd, rwd])
    return prob


def generate_exp_partial_a_reward_distribution(reward_table, normal_factor, mes_type, a_mask):
    '''
    Recursively generate factored exponentiated reward probability table on actions for belief propagtion.
    '''    
    r_min = normal_factor[0]
    r_max = normal_factor[1]
    epsilon = normal_factor[2]
    prob = []

    for atomic_action in reward_table.keys():
        raw = reward_table[atomic_action]['extra']
        rwd = min(math.exp(raw)/math.exp(r_max), 0.9999)
        if (mes_type == 'bw'):
            prob.append([0., rwd])
        else:
            prob.append([1. - rwd, rwd])
    return prob


def init_factored_policy(atomic_action_lst, horizon, uniform=True):
    '''
    Generate initial uninformative policy, represented as numpy array.
    '''
    if (uniform == True):
        num_of_valid_action = len(atomic_action_lst)
        policy = {}
        for time_step in range(0, horizon):
            policy[time_step] = {}
            action_prob = {}
            for action in atomic_action_lst:
                action_prob[action] = 1./num_of_valid_action
            policy[time_step] = action_prob

    return policy


def factored_policy_update(q_mf_approx, atomic_action_lst, horizon):
    '''
    Upgrade factored policy.
    param: q_mf_approx: mean field approximation of action distribution
    '''
    policy = {}
    for time_step in range(0, horizon):
        policy[time_step] = {}
        action_prob = {}
        for idx, action in enumerate(atomic_action_lst):
            if (q_mf_approx[time_step][idx] > 1. - 1e-9):
                action_prob[action] = 1. - 1e-9
            elif (q_mf_approx[time_step][idx] < 1e-9):
                action_prob[action] = 1e-9
            else:
                action_prob[action] = q_mf_approx[time_step][idx]
        policy[time_step] = action_prob
    return policy


def policy_hierarchy(time, dist, valid_actions, atomic_action_lst, mes_type, a_mask):
    '''
    Generate the form of factored distribution.
    '''
    prob = []
    num_of_valid_action = len(valid_actions)

    if (time == 0 and mes_type == 'fw'):
        for action in valid_actions:
            if (action == a_mask):
                prob.append(0.9999)
            else:
                prob.append(0.0001/(num_of_valid_action - 1))
    else:
        for action in atomic_action_lst:
            prob.append(dist[time][action])
    return prob


def trans_hierarchy(s_num, valid_actions, time, trans_dict, var_idx, s_vars, mes_type, a_mask, state = ''):
    '''
    Generate the form of factored distribution.
    '''
    if (s_num == 0):
        prob = []
        for action in valid_actions:
            prob.append(trans_dict[var_idx][state][action])
        return prob

    else:
        return [trans_hierarchy(s_num-1, valid_actions, time, trans_dict, var_idx, s_vars, mes_type, a_mask, state+'0'), trans_hierarchy(s_num-1, valid_actions, time, trans_dict, var_idx, s_vars, mes_type, a_mask, state+'1')]


def generate_factor_dist(time, state_vars, action_vars, valid_actions, atomic_action_lst, factor_type, variables, dist, mes_type, a_mask):
    '''
    Generate distribution for factor node in belief propagation.
    '''
    # state transition factor node
    if (factor_type == 'trans'):
        num = len(variables)
        # two '1' refer to the landing state variable and action variable
        s_num = num - 1 - 1
        s_var_idx = variables[-1].replace("t{}_".format(str(time)), '') + "'"
        dist_table0 = trans_hierarchy(s_num, valid_actions, time, dist, s_var_idx, state_vars, mes_type, a_mask)

    # reward factor node parameters:
    ### - state_vars: parent state_vars
    ### - action_vars: none
    ### - valid_actions: none
    ### - variables: normal_factor
    ### - dist: case
    ### - init_state: none

    elif(factor_type == 'partial_s_rwd'):
        s_num = len(state_vars)
        case = dist
        normal_factor = variables
        dist_table0 = generate_partial_reward_distribution(s_num, case, state_vars, normal_factor)

    elif(factor_type == 'exp_partial_s_rwd'):
        s_num = len(state_vars)
        case = dist
        normal_factor = variables
        dist_table0 = generate_exp_partial_reward_distribution(s_num, case, state_vars, normal_factor, mes_type)

    elif(factor_type == 'partial_a_rwd'):
        normal_factor = variables
        dist_table0 = generate_partial_a_reward_distribution(dist, normal_factor, mes_type, a_mask)

    elif(factor_type == 'exp_partial_a_rwd'):
        normal_factor = variables
        dist_table0 = generate_exp_partial_a_reward_distribution(dist, normal_factor, mes_type, a_mask)

    else:
        dist_table0 = policy_hierarchy(time, dist, valid_actions, atomic_action_lst, mes_type, a_mask)

    return np.array(dist_table0)


def generate_mega_rwd_dist(remaining_rwd, horizon, mes_type, count = 0):
    '''
    Generate one single cumulative reward node for debug purpose.
    '''
    if (remaining_rwd == 0):
        if (mes_type == 'bw'):
            prob = [0.0001, count/horizon]
        else:
            prob = [1. - count/horizon, count/horizon]
        return prob
    else:
        return [generate_mega_rwd_dist(remaining_rwd - 1, horizon, mes_type, count), generate_mega_rwd_dist(remaining_rwd - 1, horizon, mes_type, count + 1)]


def ary_idx(index):
    return int(index-1)

def ary_name(index):
    return int(index+1)

def reduce_to_factor_jax(state_dependency, valid_actions, atomic_action_lst, reward_dist, connect_dist, trans_prob, policy, s_vars, a_vars, horizon, mes_type, normal_factor, connect_debug, a_mask = ''):
    '''
    Generate dynamic bayesian network structure.
    '''
    init_candids = []
    policy_factors = []
    initial_factors = []

    f_actions = vgroup.NDVarArray(num_states=len(atomic_action_lst), shape=(horizon,))
    f_states = vgroup.NDVarArray(num_states=2, shape=(horizon, len(s_vars)))

    default_act = 'noop'
    len_of_cases = len(reward_dist[default_act]['parents'])
    f_p_rwds = vgroup.NDVarArray(num_states=2, shape=(horizon, len_of_cases+1))
    f_p_cumus = vgroup.NDVarArray(num_states=2, shape=(horizon, len_of_cases+1))
    
    f_rwds = vgroup.NDVarArray(num_states=2, shape=(horizon,))
    f_cumus = vgroup.NDVarArray(num_states=2, shape=(horizon+1,))
    
    fg = fgraph.FactorGraph(variable_groups=[f_actions, f_states, f_p_rwds, f_p_cumus, f_rwds, f_cumus])

    connecting_factor_vars = []
    policy_f_dists = []
    completed_factors = []
    for time_step in range(0, horizon):
        time_stamp = str(time_step)
        next_time = time_step + 1
        next_stamp = str(next_time)
        prev_time = time_step - 1
        prev_stamp = str(prev_time)

        policy_factor_vars = []
        factor_var = []
        factor_var.append("t" + time_stamp + "_atomic_action")
        policy_factor_vars.append(factor_var)
        for idx, factor_var in enumerate(policy_factor_vars):
            policy_f_dist = generate_factor_dist(time_step, s_vars, a_vars, valid_actions, atomic_action_lst, "policy", factor_var, policy, mes_type, a_mask)
            policy_f_dists.append(policy_f_dist)
        
        if (time_step != 0):
            trans_factor_vars = []
            f_var_names = []
            for cs_idx, s_var in enumerate(s_vars):
                child_var = s_var + "'"
                factor_var = []
                f_var_name = []

                for curr_s_var in state_dependency[child_var]:
                    ps_idx = s_vars.index(curr_s_var)
                    factor_var.append("t" + prev_stamp + "_" + curr_s_var)
                    f_var_name.append(f_states[prev_time, ps_idx])
                    if (time_step == 1):
                        if (not curr_s_var in init_candids):
                            init_candids.append(curr_s_var)

                factor_var.append("t" + prev_stamp + "_atomic_action")
                factor_var.append("t" + time_stamp + "_" + s_var)
                f_var_name.append(f_actions[prev_time])
                f_var_name.append(f_states[time_step, cs_idx])

                trans_factor_vars.append(factor_var)
                f_var_names.append(f_var_name)

            for landing_s, factor_var, fg_vars in zip(s_vars, trans_factor_vars, f_var_names):
                parent_s = state_dependency[landing_s + "'"]
                trans_f_dist = generate_factor_dist(time_step, parent_s, a_vars, valid_actions, atomic_action_lst, "trans", factor_var, trans_prob, mes_type, a_mask)

                s_configs = np.array(list(itertools.product(np.arange(2), repeat=len(fg_vars)-2)))
                s_dim = len(s_configs)
                s_configs = np.repeat(s_configs, len(atomic_action_lst), axis=0)
                a_configs = np.array(list(range(len(atomic_action_lst))))
                a_configs = np.tile(a_configs, (s_dim, 1))
                a_configs = a_configs.reshape((-1, 1))
                p_configs = np.append(s_configs, a_configs, axis=1)
                c_dim = len(p_configs)
                p_configs = np.repeat(p_configs, 2, axis=0)
                p_configs = p_configs.reshape((int(c_dim*2), -1))
                c_configs = np.array([0, 1])
                c_configs = np.tile(c_configs, (c_dim, 1))
                c_configs = c_configs.reshape((-1, 1))
                f_configs = np.append(p_configs, c_configs, axis=1)
                f_configs = f_configs.astype(int)

                completed_factors.append(F.enum.EnumFactor(
                    variables=fg_vars,
                    factor_configs=f_configs,
                    log_potentials=np.log(trans_f_dist.flatten()),))

        # Reformulation of rewards

        # because all action share the same reward table except for the intrinsic action cost. (we omit 2 domains that fail this assumption)
        # we only need to set one action to enumerate the state-dependent reward table
        default_act = 'noop'
        len_of_cases = len(reward_dist[default_act]['parents'])

        partial_rwd_s_factor_vars = []
        r_f_vars = []
        for idx in range(0, len_of_cases):
            factor_var = []
            r_f_var = []
            for svar in reward_dist[default_act]['parents'][idx]:
                ps_idx = s_vars.index(svar)
                factor_var.append("t" + time_stamp + "_" + svar)
                r_f_var.append(f_states[time_step, ps_idx])
                if (not svar in init_candids):
                    init_candids.append(svar)
            factor_var.append("t" + next_stamp + "_pr" + str(idx+1))
            r_f_var.append(f_p_rwds[ary_idx(next_time), idx])
            partial_rwd_s_factor_vars.append(factor_var)
            r_f_vars.append(r_f_var)

        # formalize distribution for factor nodes

        for case, vars, factor_var, r_factor in zip(reward_dist[default_act]['cases'], reward_dist[default_act]['parents'], partial_rwd_s_factor_vars, r_f_vars):
            r_f_configs = np.array(list(itertools.product(np.arange(2), repeat=len(r_factor))))
            pr_s_f_dist = generate_factor_dist(next_time, vars, None, None, None, "partial_s_rwd", normal_factor, case, mes_type, a_mask)

            completed_factors.append(F.enum.EnumFactor(
                variables=r_factor,
                factor_configs=r_f_configs,
                log_potentials=np.log(pr_s_f_dist.flatten()),))

        partial_rwd_a_factor_vars = []
        r_a_f_vars = []
        factor_var = []
        r_a_f_var = []
        factor_var.append("t" + time_stamp + "_atomic_action" )
        factor_var.append("t" + next_stamp + "_pr" + str(len_of_cases + 1))
        r_a_f_var.append(f_actions[time_step])
        r_a_f_var.append(f_p_rwds[ary_idx(next_time), len_of_cases])
        partial_rwd_a_factor_vars.append(factor_var)
        r_a_f_vars.append(r_a_f_var)
        # formalize distribution for factor nodes
        for factor_var, r_factor in zip(partial_rwd_a_factor_vars, r_a_f_vars):
            r_a_configs = np.array(list(range(len(atomic_action_lst))))
            r_a_configs = np.repeat(r_a_configs, 2, axis=0)
            r_configs = np.array([0, 1])
            r_configs = np.tile(r_configs, (len(atomic_action_lst), 1))
            r_a_configs = r_a_configs.reshape((-1, 1))
            r_configs = r_configs.reshape((-1, 1))
            f_configs = np.append(r_a_configs, r_configs, axis=1)

            pr_a_f_dist = generate_factor_dist(next_time, None, None, None, None, "partial_a_rwd", normal_factor, reward_dist, mes_type, a_mask)

            fg.add_factors(F.enum.EnumFactor(
                variables=r_factor,
                factor_configs=f_configs,
                log_potentials=np.log(pr_a_f_dist.flatten()),))

        completed_factors.append(F.enum.EnumFactor(
            variables=[(f_p_cumus[ary_idx(next_time), 0])],
            factor_configs=np.arange(2)[:, None],
            log_potentials=np.log(np.array([0.0001, 0.9999])),))
        
        cumu_rwd_factor_vars = []
        cumu_f_vars = []
        for idx in range(0, len_of_cases):
            factor_var = []
            cumu_f_var = []
            factor_var.append("t" + next_stamp + "_cr" + str(idx))
            factor_var.append("t" + next_stamp + "_pr" + str(idx+1))
            factor_var.append("t" + next_stamp + "_cr" + str(idx+1))

            cumu_f_var.append(f_p_cumus[ary_idx(next_time), idx])
            cumu_f_var.append(f_p_rwds[ary_idx(next_time), ary_idx(idx+1)])
            cumu_f_var.append(f_p_cumus[ary_idx(next_time), idx+1])
            cumu_rwd_factor_vars.append(factor_var)
            cumu_f_vars.append(cumu_f_var)
        # Final step reward
        factor_var = []
        cumu_f_var = []
        factor_var.append("t" + next_stamp + "_cr" + str(len_of_cases))
        factor_var.append("t" + next_stamp + "_pr" + str(len_of_cases+1))
        factor_var.append("r" + next_stamp)

        cumu_f_var.append(f_p_cumus[ary_idx(next_time), len_of_cases])
        cumu_f_var.append(f_p_rwds[ary_idx(next_time), ary_idx(len_of_cases+1)])
        cumu_f_var.append(f_rwds[ary_idx(next_time)])
        cumu_rwd_factor_vars.append(factor_var)
        cumu_f_vars.append(cumu_f_var)

        # formalize distribution for factor nodes
        auxiliary_dist = generate_connect_distribution(len_of_cases + 1)

        for idx, factor_var, cumu_f_var in zip(list(range(len(cumu_rwd_factor_vars))), cumu_rwd_factor_vars, cumu_f_vars):
            f_configs = np.array(list(itertools.product(np.arange(2), repeat=len(cumu_f_var))))
            aux_r_dist = np.array([[[1 - auxiliary_dist[idx+1]['00'], auxiliary_dist[idx+1]['00']], [1 - auxiliary_dist[idx+1]['01'], auxiliary_dist[idx+1]['01']]], [[1 - auxiliary_dist[idx+1]['10'], auxiliary_dist[idx+1]['10']], [1 - auxiliary_dist[idx+1]['11'], auxiliary_dist[idx+1]['11']]]])

            completed_factors.append(F.enum.EnumFactor(
                variables=cumu_f_var,
                factor_configs=f_configs,
                log_potentials=np.log(aux_r_dist.flatten()),))


        # trick to connect reward of different time slice
        connecting_factor_vars = []
        connecting_factor_vars.append("c" + time_stamp)
        connecting_factor_vars.append("r" + next_stamp)
        connecting_factor_vars.append("c" + next_stamp)

        connect_f_vars = []
        connect_f_vars.append(f_cumus[time_step])
        connect_f_vars.append(f_rwds[ary_idx(next_time)])
        connect_f_vars.append(f_cumus[next_time])

        con_f_dist_raw = np.array([[[1 - connect_dist[next_time]['00'], connect_dist[next_time]['00']], [1 - connect_dist[next_time]['01'], connect_dist[next_time]['01']]], [[1 - connect_dist[next_time]['10'], connect_dist[next_time]['10']], [1 - connect_dist[next_time]['11'], connect_dist[next_time]['11']]]])
        if (mes_type == 'bw' and next_time == horizon):
            # backward loopy bp formulation
            ob_mask = np.array([[[0.0001, 0.9999], [0.0001, 0.9999]],[[0.0001, 0.9999], [0.0001, 0.9999]]])
        else:
            ob_mask = np.array([[[1.0, 1.0], [1.0, 1.0]],[[1.0, 1.0], [1.0, 1.0]]])
        con_f_dist = np.multiply(con_f_dist_raw, ob_mask)

        f_configs = np.array(list(itertools.product(np.arange(2), repeat=len(connect_f_vars))))
        completed_factors.append(F.enum.EnumFactor(
            variables=connect_f_vars,
            factor_configs=f_configs,
            log_potentials=np.log(con_f_dist.flatten()),))

    policy_unaries = fgroup.EnumFactorGroup(
        variables_for_factors=[[f_actions[t]] for t in range(0, horizon)],
        factor_configs=np.arange(len(atomic_action_lst))[:, None],
        log_potentials=np.stack(policy_f_dists, axis=0),
    )

    completed_factors.insert(0, policy_unaries)

    completed_factors.insert(0, F.enum.EnumFactor(
        variables=[f_cumus[0]],
        factor_configs=np.arange(2)[:, None],
        log_potentials=np.log(np.array([0.0001, 0.9999])),))

    init_vs = []
    for s_var in init_candids:
        init_s_idx = s_vars.index(s_var)
        init_vs.append([f_states[0, init_s_idx]])
    init_unaries = fgroup.EnumFactorGroup(
        variables_for_factors=init_vs,
        factor_configs=np.arange(2)[:, None],
        log_potentials=np.stack([np.zeros(len(init_candids)), np.ones(len(init_candids))], axis=1),
    )

    completed_factors.insert(0, init_unaries)
    fg.add_factors(completed_factors)

    return [fg, init_candids, policy_unaries, init_unaries]


def reduce_to_factor(state_dependency, valid_actions, atomic_action_lst, reward_dist, connect_dist, trans_prob, policy, s_vars, a_vars, horizon, init_state, mes_type, normal_factor, connect_debug, a_mask = ''):
    '''
    Generate factor node in factor graph.
    '''
    init_factor_nodes = {}
    policy_factor_nodes = {}
    trans_factor_nodes = {}
    partial_reward_factor_nodes = {}
    cumu_rwd_factor_nodes = {}
    con_factor_nodes = {}
    init_candids = []

    connecting_factor_vars = []
    for time_step in range(0, horizon):
        time_stamp = str(time_step)
        next_time = time_step + 1
        next_stamp = str(next_time)
        prev_time = time_step - 1
        prev_stamp = str(prev_time)

        policy_factor_vars = []
        factor_var = []
        factor_var.append("t" + time_stamp + "_atomic_action")
        policy_factor_vars.append(factor_var)
        # formalize distribution for factor nodes
        policy_f_dists = []
        policy_f_nodes = []
        for factor_var in policy_factor_vars:
            policy_f_dist = generate_factor_dist(time_step, s_vars, a_vars, valid_actions, atomic_action_lst, "policy", factor_var, policy, mes_type, a_mask)
            policy_f_dists.append(policy_f_dist)
            policy_f_nodes.append(factor(factor_var, policy_f_dist))

        policy_factor_nodes[time_step] = policy_f_nodes

        if (time_step != 0):
            trans_factor_vars = []
            for s_var in s_vars:
                child_var = s_var + "'"
                factor_var = []
                # print(child_var)
                # print(state_dependency[child_var])
                for curr_s_var in state_dependency[child_var]:
                    factor_var.append("t" + prev_stamp + "_" + curr_s_var)
                    if (time_step == 1):
                        if (not curr_s_var in init_candids):
                            init_candids.append(curr_s_var)

                factor_var.append("t" + prev_stamp + "_atomic_action")
                factor_var.append("t" + time_stamp + "_" + s_var)
                trans_factor_vars.append(factor_var)
            # formalize distribution for factor nodes
            trans_f_dists = []
            trans_f_nodes = []
            for landing_s, factor_var in zip(s_vars, trans_factor_vars):
                parent_s = state_dependency[landing_s + "'"]
                trans_f_dist = generate_factor_dist(time_step, parent_s, a_vars, valid_actions, atomic_action_lst, "trans", factor_var, trans_prob, mes_type, a_mask)
                trans_f_nodes.append(factor(factor_var, trans_f_dist))
                trans_f_dists.append(trans_f_dist)
            trans_factor_nodes[time_step] = trans_f_nodes


        # Reformulation of rewards

        # because all action share the same reward table except for the intrinsic action cost. (we omit 2 domains that fail this assumption)
        # we only need to set one action to enumerate the state-dependent reward table
        default_act = 'noop'
        len_of_cases = len(reward_dist[default_act]['parents'])

        partial_rwd_s_factor_vars = []
        for idx in range(0, len_of_cases):
            factor_var = []
            for svar in reward_dist[default_act]['parents'][idx]:
                factor_var.append("t" + time_stamp + "_" + svar)
                init_candids.append(svar)
            factor_var.append("t" + next_stamp + "_pr" + str(idx+1))
            partial_rwd_s_factor_vars.append(factor_var)
        # formalize distribution for factor nodes
        pr_f_dists = []
        pr_f_nodes = []
        for case, vars, factor_var in zip(reward_dist[default_act]['cases'], reward_dist[default_act]['parents'], partial_rwd_s_factor_vars):
            pr_s_f_dist = generate_factor_dist(next_time, vars, None, None, None, "partial_s_rwd", normal_factor, case, mes_type, a_mask)
            pr_f_nodes.append(factor(factor_var, pr_s_f_dist))
            pr_f_dists.append(pr_s_f_dist)

        partial_rwd_a_factor_vars = []
        factor_var = []
        factor_var.append("t" + time_stamp + "_atomic_action" )
        factor_var.append("t" + next_stamp + "_pr" + str(len_of_cases + 1))
        partial_rwd_a_factor_vars.append(factor_var)
        # formalize distribution for factor nodes
        for factor_var in partial_rwd_a_factor_vars:
            pr_a_f_dist = generate_factor_dist(next_time, None, None, None, None, "partial_a_rwd", normal_factor, reward_dist, mes_type, a_mask)
            pr_f_nodes.append(factor(factor_var, pr_a_f_dist))
            pr_f_dists.append(pr_a_f_dist)
        partial_reward_factor_nodes[next_time] = pr_f_nodes

        if (connect_debug):
            cumu_rwd_factor_vars = []
            for idx in range(0, len_of_cases + 1):
                cumu_rwd_factor_vars.append("t" + next_stamp + "_pr" + str(idx+1))
            cumu_rwd_factor_vars.append("r" + next_stamp)
            aux_r_dist = np.array(generate_mega_rwd_dist(len_of_cases + 1, len_of_cases + 1, '_'))
            c_f_rwd_nodes = [factor(cumu_rwd_factor_vars, aux_r_dist)]
            cumu_rwd_factor_nodes[next_time] = c_f_rwd_nodes
        else:
            cumu_rwd_factor_vars = []
            for idx in range(0, len_of_cases):
                factor_var = []
                factor_var.append("t" + next_stamp + "_cr" + str(idx))
                factor_var.append("t" + next_stamp + "_pr" + str(idx+1))
                factor_var.append("t" + next_stamp + "_cr" + str(idx+1))
                cumu_rwd_factor_vars.append(factor_var)
            # Final step reward
            factor_var = []
            factor_var.append("t" + next_stamp + "_cr" + str(len_of_cases))
            factor_var.append("t" + next_stamp + "_pr" + str(len_of_cases+1))
            factor_var.append("r" + next_stamp)
            cumu_rwd_factor_vars.append(factor_var)
            # formalize distribution for factor nodes
            auxiliary_dist = generate_connect_distribution(len_of_cases + 1)
            c_f_rwd_dists = []
            c_f_rwd_nodes = []
            for idx, factor_var in enumerate(cumu_rwd_factor_vars):
                aux_r_dist = np.array([[[1 - auxiliary_dist[idx+1]['00'], auxiliary_dist[idx+1]['00']], [1 - auxiliary_dist[idx+1]['01'], auxiliary_dist[idx+1]['01']]], [[1 - auxiliary_dist[idx+1]['10'], auxiliary_dist[idx+1]['10']], [1 - auxiliary_dist[idx+1]['11'], auxiliary_dist[idx+1]['11']]]])
                c_f_rwd_nodes.append(factor(factor_var, aux_r_dist))
                c_f_rwd_dists.append(aux_r_dist)
            cumu_rwd_factor_nodes[next_time] = c_f_rwd_nodes

        if (connect_debug):
            connecting_factor_vars.append("r" + next_stamp)
        else:
            # trick to connect reward of different time slice
            connecting_factor_vars = []
            connecting_factor_vars.append("c" + time_stamp)
            connecting_factor_vars.append("r" + next_stamp)
            connecting_factor_vars.append("c" + next_stamp)

            con_f_dist_raw = np.array([[[1 - connect_dist[next_time]['00'], connect_dist[next_time]['00']], [1 - connect_dist[next_time]['01'], connect_dist[next_time]['01']]], [[1 - connect_dist[next_time]['10'], connect_dist[next_time]['10']], [1 - connect_dist[next_time]['11'], connect_dist[next_time]['11']]]])
            if (mes_type == 'bw' and next_time == horizon):
                # backward loopy bp formulation
                ob_mask = np.array([[[0.0001, 0.9999], [0.0001, 0.9999]],[[0.0001, 0.9999], [0.0001, 0.9999]]])
            else:
                ob_mask = np.array([[[1.0, 1.0], [1.0, 1.0]],[[1.0, 1.0], [1.0, 1.0]]])
            con_f_dist = np.multiply(con_f_dist_raw, ob_mask)
            # print('connect', connecting_factor_vars, con_f_dist)
            con_f_nodes = [factor(connecting_factor_vars, con_f_dist)]

            con_factor_nodes[next_time] = con_f_nodes


    init_f_dists = []
    init_f_nodes = []
    for s_var in init_candids:
        factor_var = ["t0_" + s_var]
        s_grounding = float(init_state[s_var])
        init_f_dist =np.array([1. - max(s_grounding - 0.0001, 0.0001), max(s_grounding - 0.0001, 0.0001)])
        init_f_dists.append(init_f_dist)
        init_f_nodes.append(factor(factor_var, init_f_dist))
    init_factor_nodes[0] = init_f_nodes

    if (connect_debug):
        connecting_factor_vars.append("c" + next_stamp)
        con_f_dist = np.array(generate_mega_rwd_dist(horizon, horizon, mes_type))
        con_f_nodes = [factor(connecting_factor_vars, con_f_dist)]
        con_factor_nodes[horizon] = con_f_nodes

    return [init_factor_nodes, policy_factor_nodes, trans_factor_nodes, partial_reward_factor_nodes, cumu_rwd_factor_nodes, con_factor_nodes]