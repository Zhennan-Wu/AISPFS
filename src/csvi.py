import numpy as np
import time as timing
import random
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from utils import factored_policy_update, record


def csvi_fwd_main(parser, vi_bwd, horizon, em_num, log_file, max_iter, conv_threshold, rwd_type):
    '''
    Implementation of the forward CSVI method.
    An iterative way of calling backward CSVI methods with policy update
    '''
    atomic_action_lst = parser.get_atomic_action_lst()
    q_action_var = copy.deepcopy(csvi_bwd_main(parser, vi_bwd,  horizon, log_file, max_iter, conv_threshold, rwd_type))

    for _ in range(0, em_num):
        new_policy = copy.deepcopy(factored_policy_update(q_action_var, atomic_action_lst, horizon))
        vi_bwd.policy_update(new_policy)
        q_action_var = copy.deepcopy(csvi_bwd_main(parser, vi_bwd, horizon, log_file, max_iter, conv_threshold, rwd_type))

    return q_action_var


def csvi_bwd_main(parser, vi_bwd, horizon, log_file, max_iter, conv_threshold, rwd_type):
    '''
    Implementation of the backward CSVI method.
    '''
    convergence = False
    iter_count = 0
    # debug
    elbo_check = False
    elbo_lst = []
    time_debug = False

    atomic_action_lst = parser.get_atomic_action_lst()

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
            if (time_debug):
                print('CSVI action update per variable use {} second, take previous approximation use {} second'.format(end - start, grasp_end - grasp_start))
            curr_action_approx = copy.deepcopy(vi_bwd.get_action_approx_dist(time_step))

            for idx, _ in enumerate(atomic_action_lst):
                diff = abs(prev_action_approx[idx] -
                           curr_action_approx[idx])
                if (diff > max_gap):
                    max_gap = diff

        if (max_gap < conv_threshold):
            convergence = True

        if (iter_count == max_iter):
            break
    if (convergence):
        if (elbo_check):
            plt.plot(elbo_lst, '*', label = 'csvi_bwd', linestyle='-', linewidth=1)
            plt.title('ELBO after each update', fontweight='bold')
            plt.xlabel("Update")
            plt.ylabel("ELBO value")
            plt.legend()
            plt.savefig('../ELBO/csvi_bwd_{}_elbo.png'.format(random.random()))
            plt.close()
        else:
            pass
    else:
        record("CSVI Backwards Convergence Failed in {} iterations".format(
            max_iter), log_file)
    return vi_bwd.get_full_action_approx_dist()