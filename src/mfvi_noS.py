import numpy as np
import time as timing
import matplotlib.pyplot as plt
import random
import copy
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from variational_inf import Variational_inference
from utils import *


def mfvi_fwd_noS_main(parser, vi_bwd, horizon, em_num, log_file, max_iter, conv_threshold):
    '''
    Implementation of the forward MFVI method without update of state variables.
    '''
    atomic_action_lst = parser.get_atomic_action_lst()

    q_action_var = copy.deepcopy(mfvi_bwd_noS_main(parser, vi_bwd, horizon, log_file, max_iter, conv_threshold))

    for _ in range(0, em_num):
        new_policy = copy.deepcopy(factored_policy_update(q_action_var, atomic_action_lst, horizon))
        vi_bwd.policy_update(new_policy)
        q_action_var = copy.deepcopy(mfvi_bwd_noS_main(parser, vi_bwd, horizon, log_file, max_iter, conv_threshold))

    return q_action_var


def mfvi_bwd_noS_main(parser, vi_bwd, horizon, log_file, max_iter, conv_threshold, rwd_type = 'std'):
    '''
    Implementation of the backward MFVI method without update of state variables.
    '''
    num_of_rwd = vi_bwd.get_num_of_rwd_case()
    a_conv = False
    r_conv = False
    c_conv = False
    convergence = False
    iter_count = 0

    state_vars = parser.get_state_vars()
    atomic_action_lst = parser.get_atomic_action_lst()

    # debug
    jointly_update = True
    time_debug = False
    elbo_lst = []
    elbo_check = False
    state_debug = False

    if (horizon == 1):
        c_conv = True

    while (not convergence):
        max_a_gap = 0
        max_r_gap = 0
        max_c_gap = 0

        iter_count += 1
        for time_step in range(0, horizon):
            time_for_reward = time_step + 1
            if (jointly_update or not a_conv):
                grasp_start = timing.time()
                prev_action_approx = copy.deepcopy(vi_bwd.get_action_approx_dist(
                    time_step))
                grasp_end = timing.time()
                start = timing.time()
                vi_bwd.update_vec_action(time_step, rwd_type)
                if (elbo_check):
                    elbo = vi_bwd.calc_elbo('mfvi_bwd', rwd_type)
                    if (len(elbo_lst) > 0 and (elbo + 3) < elbo_lst[-1]):
                        print("Action update elbo not increasing in time {}, {}, {}".format(time_step, elbo_lst, elbo))
                    elbo_lst.append(elbo)
                curr_action_approx = copy.deepcopy(vi_bwd.get_action_approx_dist(time_step))
                end = timing.time()
                if (time_debug):
                    print('VI action update per variable use {} second, take previous approximation use {} second'.format(end - start, grasp_end - grasp_start))

                for a_idx, act in enumerate(atomic_action_lst):
                    diff = abs(prev_action_approx[a_idx] -
                            curr_action_approx[a_idx])
                    if (diff > max_a_gap):
                        max_a_gap = diff

            if (jointly_update or not r_conv):           
                for r_idx in range(num_of_rwd):
                    var_name = 'pr'+str(r_idx+1)
                    prev_reward_approx = copy.deepcopy(vi_bwd.get_reward_approx_dist(time_for_reward, var_name))
                    start = timing.time()
                    vi_bwd.update_vec_rwd(time_for_reward, var_name, "rwd", rwd_type)
                    if (elbo_check):
                        elbo = vi_bwd.calc_elbo('mfvi_bwd', rwd_type)
                        if (len(elbo_lst) > 0 and (elbo + 3) < elbo_lst[-1]):
                            print("Reward update elbo not increasing, {}, {}".format(elbo_lst, elbo))
                        elbo_lst.append(elbo)
                    end = timing.time()
                    if (time_debug):
                        print('VI reward variable update use {} second'.format(end - start))
                    curr_reward_approx = copy.deepcopy(vi_bwd.get_reward_approx_dist(time_for_reward, var_name))

                    diff = abs(prev_reward_approx -
                            curr_reward_approx)
                    if (diff > max_r_gap):
                        max_r_gap = diff

                    if (r_idx == num_of_rwd-1):
                        var_name = 'r'+str(time_for_reward)
                        var_type = 'final'
                    else:
                        var_name = 'pc'+str(r_idx+1)
                        var_type = 'cumu'
                    prev_reward_approx = copy.deepcopy(vi_bwd.get_reward_approx_dist(time_for_reward, var_name))
                    start = timing.time()
                    vi_bwd.update_vec_rwd(time_for_reward, var_name, var_type, rwd_type)
                    if (elbo_check):
                        elbo = vi_bwd.calc_elbo('mfvi_bwd', rwd_type)
                        if (len(elbo_lst) > 0 and (elbo + 3) < elbo_lst[-1]):
                            print("Reward update elbo not increasing, {}, {}".format(elbo_lst, elbo))
                        elbo_lst.append(elbo)
                    end = timing.time()
                    if (time_debug):
                        print('MFVI reward variable update use {} second'.format(end - start))
                    curr_reward_approx = copy.deepcopy(vi_bwd.get_reward_approx_dist(time_for_reward, var_name))

                    diff = abs(prev_reward_approx -
                            curr_reward_approx)
                    if (diff > max_r_gap):
                        max_r_gap = diff

            if (jointly_update or not c_conv):
                if (time_for_reward < horizon):
                    c_idx = 'c'+str(time_for_reward)
                    prev_connect_approx = copy.deepcopy(vi_bwd.get_connect_approx_dist(c_idx))
                    start = timing.time()
                    vi_bwd.update_vec_cumu(time_for_reward, c_idx, rwd_type)
                    if (elbo_check):
                        elbo = vi_bwd.calc_elbo('mfvi_bwd', rwd_type)
                        if (len(elbo_lst) > 0 and (elbo + 3) < elbo_lst[-1]):
                            print("Connect update elbo not increasing, {}, {}".format(elbo_lst, elbo))
                        elbo_lst.append(elbo)
                    end = timing.time()
                    if (time_debug):
                        print('VI cumulative variable update use {} second'.format(end - start))
                    curr_connect_approx = copy.deepcopy(vi_bwd.get_connect_approx_dist(c_idx))

                    diff = abs(prev_connect_approx -
                            curr_connect_approx)
                    if (diff > max_c_gap):
                        max_c_gap = diff

        if (max_a_gap < conv_threshold):
            a_conv = True
        else:
            a_conv = False

        if (max_r_gap < conv_threshold):
            r_conv = True
        else:
            r_conv = False

        if (max_c_gap < conv_threshold):
            c_conv = True
        else:
            c_conv = False

        if (a_conv and r_conv and c_conv):
            convergence = True

        if (iter_count == max_iter):
            break

    if (convergence):
        if (elbo_check):
            plt.plot(elbo_lst, 'o', label ='mfvi_bwd', linestyle='-', linewidth=1)
            plt.title('ELBO after each update', fontweight="bold", fontsize=18)
            plt.xlabel("Update", fontweight="bold", fontsize=15)
            plt.ylabel("ELBO value", fontweight="bold", fontsize=15)
            plt.legend(prop={'weight':'bold', 'size': 14})
            plt.tick_params(axis='x', labelsize=14)
            plt.tick_params(axis='y', labelsize=14)
            plt.savefig('../ELBO/mfvi_bwd_{}_elbo.png'.format(random.random()),bbox_inches='tight')
            plt.close()
        else:
            pass

    else:
        record("MFVI Backwards NoS Inference Convergence Failed in {} iterations".format(
            max_iter), log_file)

    if (state_debug):
        # calculate state approximation afterwards
        after_s_conv = False
        after_max_s_gap = 0
        after_s_count = 0
        while(not after_s_conv and after_s_count < max_iter):
            after_s_count += 1
            for time_step in range(0, horizon):
                if (time_step > 0):
                    for state_var_index in state_vars:
                        grasp_start = timing.time()
                        prev_state_approx = copy.deepcopy(vi_bwd.get_state_approx_dist(
                            time_step, state_var_index))
                        grasp_end = timing.time()
                        start = timing.time()
                        vi_bwd.update_vec_state(time_step, state_var_index, rwd_type)
                        if (elbo_check):
                            elbo = vi_bwd.calc_elbo('mfvi_bwd', rwd_type)
                            if (len(elbo_lst) > 0 and (elbo + 3) < elbo_lst[-1]):
                                print("State update elbo not increasing, {}, {}".format(elbo_lst, elbo))
                            elbo_lst.append(elbo)
                        curr_state_approx = copy.deepcopy(vi_bwd.get_state_approx_dist(time_step, state_var_index))
                        end = timing.time()
                        if (time_debug):
                            print('MFVI post state update per variable use {} second, take previous approximation use {} second'.format(end - start, grasp_end - grasp_start))
                        diff = abs(
                            prev_state_approx - curr_state_approx)
                        if (diff > after_max_s_gap):
                            after_max_s_gap = diff

            if (after_max_s_gap < conv_threshold):
                after_s_conv = True
        for t in range(1, horizon):
            print_dist = {}
            for state_var_index in state_vars:
                print_dist[state_var_index] = vi_bwd.get_state_approx_dist(
                    t, state_var_index)
            print("At time {} state distribution is".format(t))
            for key in print_dist.keys():
                print(key, "            ", print_dist[key])

    return vi_bwd.get_full_action_approx_dist()