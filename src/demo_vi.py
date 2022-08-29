import numpy as np
import time as timing
import copy
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('seaborn')

from utils import record


def demo_mfvi_bwd_main(parser, vi_bwd, horizon, log_file, elbo_check, t_idx, version, domain, max_iter, conv_threshold, rwd_type='std'):
    '''
    Implementation of the backward MFVI method on demo problems, with different
    parameter corresponding to different update scheme.
    The implementation is the same as the main backward MFVI method but with detailed intermediate output.
    '''
    num_of_rwd = vi_bwd.get_num_of_rwd_case()
    convergence = False
    iter_count = 0
    state_vars = parser.get_state_vars()
    atomic_action_lst = parser.get_atomic_action_lst()

    time_debug = False
    s_conv = False
    a_conv = False
    r_conv = False
    c_conv = False
    main_s_conv = False
    skip_state = False
    elbo_lst = []
    elbo_footstep = []
    if (elbo_check):
        init_elbo = vi_bwd.calc_elbo('mfvi_bwd', 'std')
        elbo_lst.append(init_elbo)
    action_elbo_contri = 0.
    first_action_elbo_contri = 0.
    state_elbo_contri = 0.
    reward_elbo_contri = 0.
    connect_elbo_contri = 0.
    action_elbo_contri_lst = [0.]
    first_action_elbo_contri_lst = [0.]
    state_elbo_contri_lst = [0.]
    reward_elbo_contri_lst = [0.]
    connect_elbo_contri_lst = [0.]
    action_elbo_contri_dynamic = [0.]
    first_action_elbo_contri_dynamic = [0.]
    state_elbo_contri_dynamic = [0.]
    reward_elbo_contri_dynamic = [0.]
    connect_elbo_contri_dynamic = [0.]

    if (version == 'main_all'):
        skip_s_vars = []
        main_s_vars = ['proficiencyMed__s0', 'proficiencyMed__s1']
    elif (version == 'med'):
        skip_s_vars = ['updateTurn__s0', 'updateTurn__s1', 'answeredRight__s0', 'answeredRight__s1', 'proficiencyHigh__s0', 'proficiencyHigh__s1']
        main_s_vars = []
        main_s_conv = True
    elif (version == 'all'):
        skip_s_vars = []
        main_s_vars = []
        main_s_conv = True
    elif (version == 'none'):
        skip_state = True
        main_s_vars = []
        main_s_conv = True
    else:
        raise Exception('version parameter out of range')

    printed = False

    if (horizon == 1):
        s_conv = True
        main_s_conv = True
        c_conv = True

    while (not convergence):
        max_s_gap = 0
        max_main_s_gap = 0
        max_a_gap = 0
        max_r_gap = 0
        max_c_gap = 0
        action_elbo_dynamic = 0.
        first_action_elbo_dynamic = 0.
        state_elbo_dynamic = 0.
        reward_elbo_dynamic = 0.
        connect_elbo_dynamic = 0.

        iter_count += 1
        for time_step in range(0, horizon):
            time_for_reward = time_step + 1
            if (time_step > 0):
                # state update
                if (main_s_conv):
                    if (not skip_state):
                        for state_var_index in state_vars:
                            if (state_var_index not in skip_s_vars):
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
                                    elbo_footstep.append('{}_{}'.format(state_var_index, time_step))
                                    state_elbo_contri += elbo_lst[-1] - elbo_lst[-2]
                                    state_elbo_contri_lst.append(state_elbo_contri)
                                    state_elbo_dynamic += elbo_lst[-1] - elbo_lst[-2]
                                curr_state_approx = copy.deepcopy(vi_bwd.get_state_approx_dist(time_step, state_var_index))
                                end = timing.time()
                                diff = abs(
                                    prev_state_approx - curr_state_approx)
                                if (diff > max_s_gap):
                                    max_s_gap = diff
                else:
                    max_s_gap = 1
                    for state_var_index in main_s_vars:
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
                            elbo_footstep.append('{}_{}'.format(state_var_index, time_step))
                            state_elbo_contri += elbo_lst[-1] - elbo_lst[-2]
                            state_elbo_contri_lst.append(state_elbo_contri)
                            state_elbo_dynamic += elbo_lst[-1] - elbo_lst[-2]
                        curr_state_approx = copy.deepcopy(vi_bwd.get_state_approx_dist(time_step, state_var_index))
                        end = timing.time()
                        diff = abs(
                            prev_state_approx - curr_state_approx)
                        if (diff > max_main_s_gap):
                            max_main_s_gap = diff

            # action update
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
                elbo_footstep.append('a_{}'.format(time_step))
                action_elbo_contri += elbo_lst[-1] - elbo_lst[-2]
                action_elbo_contri_lst.append(action_elbo_contri)
                action_elbo_dynamic += elbo_lst[-1] - elbo_lst[-2]
                if (time_step == 0):
                    first_action_elbo_contri += elbo_lst[-1] - elbo_lst[-2]
                    first_action_elbo_contri_lst.append(first_action_elbo_contri)
                    first_action_elbo_dynamic += elbo_lst[-1] - elbo_lst[-2]
            curr_action_approx = copy.deepcopy(vi_bwd.get_action_approx_dist(time_step))
            end = timing.time()

            for a_idx, act in enumerate(atomic_action_lst):
                diff = abs(prev_action_approx[a_idx] -
                        curr_action_approx[a_idx])
                if (diff > max_a_gap):
                    max_a_gap = diff

            # reward update
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
                    elbo_footstep.append('r_{}'.format(time_for_reward))
                    reward_elbo_contri += elbo_lst[-1] - elbo_lst[-2]
                    reward_elbo_contri_lst.append(reward_elbo_contri)
                    reward_elbo_dynamic += elbo_lst[-1] - elbo_lst[-2]           
                end = timing.time()
                if (time_debug):
                    print('MFVI reward variable update use {} second'.format(end - start))
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
                    elbo_footstep.append('r_{}'.format(time_for_reward))
                    reward_elbo_contri += elbo_lst[-1] - elbo_lst[-2]
                    reward_elbo_contri_lst.append(reward_elbo_contri)
                    reward_elbo_dynamic += elbo_lst[-1] - elbo_lst[-2]
                end = timing.time()
                if (time_debug):
                    print('MFVI reward variable update use {} second'.format(end - start))
                curr_reward_approx = copy.deepcopy(vi_bwd.get_reward_approx_dist(time_for_reward, var_name))

                diff = abs(prev_reward_approx -
                        curr_reward_approx)
                if (diff > max_r_gap):
                    max_r_gap = diff

            # cumulative reward update
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
                    elbo_footstep.append('c_{}'.format(time_for_reward))
                    connect_elbo_contri += elbo_lst[-1] - elbo_lst[-2]
                    connect_elbo_contri_lst.append(connect_elbo_contri)
                    connect_elbo_dynamic += elbo_lst[-1] - elbo_lst[-2] 
                end = timing.time()
                if (time_debug):
                    print('MFVI cumulative variable update use {} second'.format(end - start))
                curr_connect_approx = copy.deepcopy(vi_bwd.get_connect_approx_dist(c_idx))

                diff = abs(prev_connect_approx -
                        curr_connect_approx)
                if (diff > max_c_gap):
                    max_c_gap = diff

        action_elbo_contri_dynamic.append(action_elbo_dynamic)
        first_action_elbo_contri_dynamic.append(first_action_elbo_dynamic)
        state_elbo_contri_dynamic.append(state_elbo_dynamic)
        reward_elbo_contri_dynamic.append(reward_elbo_dynamic)
        connect_elbo_contri_dynamic.append(connect_elbo_dynamic)
        if (max_s_gap < conv_threshold):
            s_conv = True
        else:
            s_conv = False

        if (max_main_s_gap < conv_threshold):
            max_s_gap = 0
            main_s_conv = True
        else:
            main_s_conv = False

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

        if (s_conv and a_conv and r_conv and c_conv):
            convergence = True
            if (not printed):
                for t in range(1, horizon):
                    print_dist = {}
                    for state_var_index in state_vars:
                        print_dist[state_var_index] = vi_bwd.get_state_approx_dist(
                            t, state_var_index)
                    record("At time {} state distribution is".format(t), log_file)
                    for key in print_dist.keys():
                        record('{} {} {}'.format(key, "            ", print_dist[key]), log_file)
                printed = True

        if (iter_count == max_iter):
            break

    if (convergence):
        if (elbo_check):
            log_dir = '../results/demo/elbo/' + domain[:-6] + '/' + version
            if (not os.path.isdir(log_dir)):
                os.makedirs(log_dir, exist_ok=True)
            name_map = {'none': "MFVI-NoS", "all": "MFVI", "med": "MFVI-Med", "main_all": "MFVI-Med-All"}

            # elbo composition plot
            elbo_ary = np.asarray(elbo_lst)
            plt.plot(elbo_ary, '.', label ='total elbo', linestyle='-', linewidth=1)
            action_elbo_abs_ary = np.asarray(action_elbo_contri_lst) + init_elbo
            plt.plot(action_elbo_abs_ary, '.', label ='action elbo', linestyle='-', linewidth=1)
            state_elbo_abs_ary = np.asarray(state_elbo_contri_lst) + init_elbo
            plt.plot(state_elbo_abs_ary, '.', label ='state elbo', linestyle='-', linewidth=1)
            reward_elbo_abs_ary = np.asarray(reward_elbo_contri_lst) + init_elbo
            plt.plot(reward_elbo_abs_ary, '.', label ='reward elbo', linestyle='-', linewidth=1)
            connect_elbo_abs_ary = np.asarray(connect_elbo_contri_lst) + init_elbo
            plt.plot(connect_elbo_abs_ary, '.', label ='connect elbo', linestyle='-', linewidth=1)
            plt.title('{} ELBO Increase Composition'.format(name_map[version]), fontweight="bold", fontsize=18)
            plt.xlabel("Update Iteration", fontweight="bold", fontsize=15)
            plt.ylabel("ELBO increase", fontweight="bold", fontsize=15)
            plt.legend(prop={'weight':'bold', 'size': 14})
            plt.tick_params(axis='x', labelsize=14)
            plt.tick_params(axis='y', labelsize=14)            
            plt.savefig(log_dir + '/mfvi_bwd_{}_{}_elbo_comp.png'.format(version, t_idx),bbox_inches='tight')
            plt.savefig(log_dir + '/mfvi_bwd_{}_{}_elbo_comp.pdf'.format(version, t_idx),bbox_inches='tight')
            plt.close()

            # elbo dynamic composition plot
            plt.plot(action_elbo_contri_dynamic, '*', label ='action', linestyle='-', linewidth=1)
            plt.plot(state_elbo_contri_dynamic, '*', label ='state', linestyle='-', linewidth=1)
            plt.plot(first_action_elbo_contri_dynamic, '*', label ='first action', linestyle='-', linewidth=1)
            plt.plot(reward_elbo_contri_dynamic, '*', label ='reward', linestyle='-', linewidth=1)
            plt.plot(connect_elbo_contri_dynamic, '*', label ='connect', linestyle='-', linewidth=1)
            plt.title('{} ELBO Increase Dynamic Comparison'.format(name_map[version]), fontweight="bold", fontsize=18)
            plt.xlabel("Iteration index", fontweight="bold", fontsize=15)
            plt.ylabel("ELBO increase value", fontweight="bold", fontsize=15)
            plt.legend(prop={'weight':'bold', 'size': 14})
            plt.tick_params(axis='x', labelsize=14)
            plt.tick_params(axis='y', labelsize=14)            
            plt.savefig(log_dir + '/mfvi_bwd_{}_{}_elbo_dynamic_comp.png'.format(version, t_idx),bbox_inches='tight')
            plt.savefig(log_dir + '/mfvi_bwd_{}_{}_elbo_dynamic_comp.pdf'.format(version, t_idx),bbox_inches='tight')
            plt.close()

        else:
            pass
    else:
        record("Backward Variational Inference Convergence Failed in {} iterations".format(
            max_iter), log_file)
    return vi_bwd.get_full_action_approx_dist()


def demo_csvi_bwd_main(parser, vi_bwd, horizon, log_file, elbo_check, t_idx, domain, max_iter, conv_threshold, rwd_type='std'):
    '''
    Implementation of the backward CSVI method on demo problems.
    The implementation is the same as the backward CSVI main method
    but with detailed intermediate output.
    '''
    convergence = False
    iter_count = 0
    elbo_check = True
    elbo_lst = []
    atomic_action_lst = parser.get_atomic_action_lst()
    if (elbo_check):
        elbo_lst.append(vi_bwd.calc_elbo('csvi_bwd', rwd_type))

    while (not convergence):
        max_gap = 0
        iter_count += 1
        for time_step in range(0, horizon):
            grasp_start = timing.time()
            prev_action_approx = copy.deepcopy(vi_bwd.get_action_approx_dist(
                time_step))
            grasp_end = timing.time()
            start = timing.time()
            vi_bwd.approx_a_update(time_step, rwd_type)
            if (elbo_check):
                elbo = vi_bwd.calc_elbo('csvi_bwd', rwd_type)
                if (len(elbo_lst) > 0 and (elbo + 3) < elbo_lst[-1]):
                    print("Action update elbo not increasing, {}, {}".format(elbo_lst, elbo))
                elbo_lst.append(elbo)
            end = timing.time()
            curr_action_approx = copy.deepcopy(vi_bwd.get_action_approx_dist(time_step))
            for a_idx, _ in enumerate(atomic_action_lst):
                diff = abs(prev_action_approx[a_idx] -
                           curr_action_approx[a_idx])
                if (diff > max_gap):
                    max_gap = diff

        if (max_gap < conv_threshold):
            convergence = True

        if (iter_count == max_iter):
            break
    if (convergence):
        if (elbo_check):
            log_dir = '../results/demo/elbo/' + domain[:-6] + '/csvi'
            if (not os.path.isdir(log_dir)):
                os.makedirs(log_dir, exist_ok=True)
            plt.plot(elbo_lst, '*', label = 'csvi_bwd', linestyle='-', linewidth=1)
            plt.title('ELBO after each update', fontweight="bold", fontsize=18)
            plt.xlabel("Update", fontweight="bold", fontsize=15)
            plt.ylabel("ELBO value", fontweight="bold", fontsize=15)
            plt.legend(prop={'weight':'bold', 'size': 14})
            plt.tick_params(axis='x', labelsize=14)
            plt.tick_params(axis='y', labelsize=14)            
            plt.savefig(log_dir + '/csvi_bwd_{}_elbo.png'.format(t_idx),bbox_inches='tight')
            plt.savefig(log_dir + '/csvi_bwd_{}_elbo.pdf'.format(t_idx), bbox_inches='tight')
            plt.close()
        else:
            pass
    else:
        record("CSVI Convergence Failed in {} iterations".format(
            max_iter), log_file)
    return vi_bwd.get_full_action_approx_dist()