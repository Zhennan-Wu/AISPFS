import math
import random
import numpy as np
import time as timing
import copy
import json
import configargparse

from spudd_parser import SPUDD_Parser
from utils import generate_same_time_joint_cumu, generate_cross_time_joint_cumu, init_factored_policy

class Variational_inference():

    def __init__(self, parser, horizon, param_path):
        self.v = ("0", "1")
        self.state_vars = parser.get_state_vars()
        self.num_of_svars = len(self.state_vars)
        self.rwd_cases = parser.get_reward_cases()
        self.atomic_action_lst = parser.get_atomic_action_lst()
        self.num_of_action = len(self.atomic_action_lst)
        self.joint_trans = parser.get_joint_trans()
        self.r_min, self.r_max, self.epsilon = parser.get_normal_factor()
        self.r_min_unf, self.r_max_unf = parser.get_unfactored_bound()
        # a list of matrix that multiply with full state will output relevant state variables value of the compact transition matrix for each state variable
        trans_factor = []
        # a list of np.array of power of 2, multiplied with state array will produce the state index in the transition matrix
        trans_idx= []
        # dimensition of the concatenated matrix of all state vars transition
        idx_matrix_dim = 0
        case_num = 0
        # a list of parent index in the matrix for each state vars
        trans_break_point = []
        for case in self.joint_trans:
            case_num += 1
            # generate matrix of dimensition len(relevant state vars)*len(full state vars)
            trans_factor.append(self.generate_factor_matrix(case['row_var'][:-1]))

            # calculate index in transiton matrix
            # row vars contain children var, so need to substract 1
            vec_dim = len(case['row_var']) - 1
            if (vec_dim == 0):
                next_dim = idx_matrix_dim + 1
            else:
                next_dim = idx_matrix_dim + vec_dim
            trans_break_point.append([idx_matrix_dim, next_dim])
            idx_matrix_dim = next_dim

            if (vec_dim == 0):
                # for isolated state variables
                idx_base = [0]
            else:
                # add 1 because the transition matrix has child variable as the last variable
                idx_base = [pow(2, i+1) for i in range(vec_dim)]
            idx_base.reverse()

            trans_idx.append(idx_base)
        # a matrix of dimension state variable * total dependent state variable (duplicates), multiply with concatenated dependent state will produce state index in compact transition
        self.trans_idx_matrix = np.zeros((case_num, idx_matrix_dim))
        self.trans_factor_matrix = np.concatenate(trans_factor, 0)
        case_idx = 0
        for bp, idx in zip(trans_break_point, trans_idx):
            self.trans_idx_matrix[case_idx][bp[0]:bp[1]] = np.array(idx)
            case_idx += 1

        self.joint_rwd = parser.get_joint_rwd()
        self.joint_raw_rwd = parser.get_joint_raw_rwd()
        self.joint_exp_rwd = parser.get_joint_exp_rwd()
        
        # components for faster CSVI
        rwd_factor = []
        rwd_idx = []
        idx_matrix_dim = 0
        case_num = 0

        rwd_break_point = []
        for case in self.joint_rwd:
            case_num += 1
            rwd_factor.append(self.generate_factor_matrix(case['row_var'][:-1]))
            # calculate index in transiton matrix
            vec_dim = len(case['row_var']) - 1
            if (vec_dim == 0):
                next_dim = idx_matrix_dim + 1
            else:
                next_dim = idx_matrix_dim + vec_dim
            rwd_break_point.append([idx_matrix_dim, next_dim])
            idx_matrix_dim = next_dim
            # for reward on actions
            if (vec_dim == 0):
                idx_base = [0]
            # for reward on state variable
            else:
                idx_base = [pow(2, i+1) for i in range(vec_dim)]
            idx_base.reverse()
            rwd_idx.append(idx_base)
        # add 1 for reward on actions
        self.rwd_idx_matrix = np.zeros((case_num, idx_matrix_dim))
        # a matrix of #cases * #state vars
        self.rwd_factor_matrix = np.concatenate(rwd_factor, 0)
        case_idx = 0
        for bp, idx in zip(rwd_break_point, rwd_idx):
            self.rwd_idx_matrix[case_idx][bp[0]:bp[1]] = np.array(idx)
            case_idx += 1
            
        self.same_time_joint_cumu, self.same_time_cumu_group = generate_same_time_joint_cumu(self.rwd_cases)
        self.cross_time_joint_cumu, self.cross_time_cumu_group = generate_cross_time_joint_cumu(horizon)
        self.trans_group = parser.get_trans_group()
        self.rwd_group = parser.get_rwd_group()
        self.trans = parser.get_trans()
        self.init_state = {}

        self.h = horizon
        self.details = False
        self.vec_trans = parser.get_vec_trans()
        # sample size
        with open(param_path, 'r') as params_file:
            params = json.load(params_file)
            self.a_seq_size = params['act_seq_sample_size']
            self.full_traj_sample_size = params['full_traj_sample_size']

        # initialize approximate distributions
        self.full_sample_size = self.a_seq_size * self.full_traj_sample_size
        self.policy = None
        self.approx_state_dist = {}
        self.approx_action_dist = {}
        self.approx_reward_dist = {}
        self.approx_connect_dist = {}

    def generate_factor_matrix(self, vars):
        '''
        generate a matrix which extract relevant state vars from all state vars
        '''
        if (len(vars) > 0):
            m = np.zeros((len(vars), self.num_of_svars))
            for idx, var in enumerate(vars):
                m[idx, self.state_vars.index(var)] = 1.
        else:
            # for reward on actions
            m = np.array([np.zeros(self.num_of_svars)])
        return m
        
    def init_approx(self, init_state, policy, h):
        h = min(self.h, h)
        if (h != self.h):
            self.cross_time_joint_cumu, self.cross_time_cumu_group = generate_cross_time_joint_cumu(h)
            self.h = h

        self.policy_update(policy)
        self.init_state = init_state

        for time_step in range(0, h+1):
            if (time_step > 0 and time_step < h):
                state_approx = {}
                for var in self.state_vars:
                    state_approx[var] = 0.5
                self.approx_state_dist[time_step] = state_approx
                self.approx_reward_dist[time_step] = {}
                self.approx_reward_dist[time_step]['pc0'] = 1.
                for c in range(self.rwd_cases):
                    self.approx_reward_dist[time_step]['pr' + str(c+1)] = 0.5
                    if (c == self.rwd_cases - 1):
                        self.approx_reward_dist[time_step]['r'+str(time_step)] = 0.5
                    else:
                        self.approx_reward_dist[time_step]['pc' + str(c+1)] = 0.5

            elif (time_step == 0):
                state_approx = {}
                for var in self.state_vars:
                    val = init_state[var]
                    if (val == '0'):
                        state_approx[var] = 0.
                    elif (val == '1'):
                        state_approx[var] = 1.
                    else: 
                        raise Exception("init_state format wrong", init_state)
                self.approx_state_dist[time_step] = state_approx
                self.approx_reward_dist[time_step] = {}
                self.approx_reward_dist[time_step]['pc0'] = 1.
                for c in range(self.rwd_cases):
                    self.approx_reward_dist[time_step]['pr' + str(c+1)] = 0.5
                    if (c == self.rwd_cases - 1):
                        self.approx_reward_dist[time_step]['r'+str(time_step)] = 0.5
                    else:
                        self.approx_reward_dist[time_step]['pc' + str(c+1)] = 0.5
            else:
                state_approx = {}
                for var in self.state_vars:
                    state_approx[var] = 0.5
                self.approx_state_dist[time_step] = state_approx
                self.approx_reward_dist[time_step] = {}
                self.approx_reward_dist[time_step]['pc0'] = 1.
                for c in range(self.rwd_cases):
                    self.approx_reward_dist[time_step]['pr' + str(c+1)] = 0.5
                    if (c == self.rwd_cases - 1):
                        self.approx_reward_dist[time_step]['r'+str(time_step)] = 0.5
                    else:
                        self.approx_reward_dist[time_step]['pc' + str(c+1)] = 0.5

            if (time_step < h):
                valid_act_num = len(self.atomic_action_lst)
                action_approx = [1./valid_act_num]*valid_act_num
                self.approx_action_dist[time_step] = action_approx

            if (time_step < 1):
                self.approx_connect_dist['c'+str(time_step)] = 0.9999
            elif (time_step == h):
                self.approx_connect_dist['c'+str(time_step)] = 1.
            else:
                self.approx_connect_dist['c'+str(time_step)] = 0.5

    def policy_update(self, policy):
        self.policy = policy

    def get_num_of_rwd_case(self):
        return self.rwd_cases

    def get_reward_approx_dist(self, time, var_name):
        return self.approx_reward_dist[time][var_name]

    def get_connect_approx_dist(self, var_name):
        return self.approx_connect_dist[var_name]

    def get_action_approx_dist(self, time):
        return self.approx_action_dist[time]

    def get_full_action_approx_dist(self):
        return self.approx_action_dist

    def get_state_approx_dist(self, time, var_idx):
        return self.approx_state_dist[time][var_idx]

    def vec_policy(self, time):
        return np.array(list(self.policy[time].values()))

    def generate_approx(self, time, var_name, var_type, case_id, case_type, rwd_type):
        '''
        For reward function, time is the reward variable index
        For transition function, time is the current state and action time step, i.e. for the child state to be the considered variable, the time parameter should be the previous time step
        '''
        truth = None
        if (case_type == 'rwd'):
            if (rwd_type == 'exp'):
                truth = self.joint_exp_rwd[case_id]
            elif (rwd_type == 'std'):
                truth = self.joint_rwd[case_id]
            else:
                raise Exception('Uncaptured reward type')
        elif (case_type == 'trans'):
            truth = self.joint_trans[case_id]
        elif (case_type == 'same_time_cumu'):
            truth = self.same_time_joint_cumu[case_id]
        elif (case_type == 'cross_time_cumu'):
            truth = self.cross_time_joint_cumu[case_id]
        else:
            raise Exception('Uncaptured case')
        # generate approximate matrix of the same shape and positioning of the true matrix
        if (var_type =='elbo'):
            if (case_type == 'same_time_cumu'):
                dist = np.array(1)
                for var in reversed(truth['row_var']):
                    if (var == 'r'):
                        var = var + str(time)
                    if (var == 'pc0'):
                        prob = np.array([1., 1.])
                    else:
                        prob = np.array([1. - self.approx_reward_dist[time][var], self.approx_reward_dist[time][var]])

                    dist = np.outer(prob, dist).flatten()
                approx_matrix = dist

            elif (case_type == 'cross_time_cumu'):
                dist = np.array(1)
                for var in reversed(truth['row_var']):
                    if ('c' in var):
                        if (var == 'c0'):
                            prob = np.array([1., 1.])
                        else:
                            prob = np.array([1. - self.approx_connect_dist[var], self.approx_connect_dist[var]])
                    else:
                        var_t = int(var[1])
                        prob = np.array([1. - self.approx_reward_dist[var_t][var], self.approx_reward_dist[var_t][var]])

                    dist = np.outer(prob, dist).flatten()
                approx_matrix = dist

            elif (case_type == 'rwd'):
                dist = np.array(1)
                for var in reversed(truth['row_var']):
                    if ('pr' in var and var[2:].isnumeric()):
                        prob = np.array([1. - self.approx_reward_dist[time][var], self.approx_reward_dist[time][var]])
                    else:
                        prob = np.array([1. - self.approx_state_dist[time-1][var], self.approx_state_dist[time-1][var]])

                    dist = np.outer(prob, dist).flatten()

                act_prob = np.array(self.approx_action_dist[time-1])
                approx_matrix = np.outer(dist, act_prob)

            elif (case_type == 'trans'):
                dist = np.array(1)
                for var in reversed(truth['row_var']):
                    if (var[-1] == "'"):
                        var = var.replace("'", "")
                        prob = np.array([1. - self.approx_state_dist[time][var], self.approx_state_dist[time][var]])
                    else:
                        prob = np.array([1. - self.approx_state_dist[time-1][var], self.approx_state_dist[time-1][var]])

                    dist = np.outer(prob, dist).flatten()

                act_prob = np.array(self.approx_action_dist[time-1])
                approx_matrix = np.outer(dist, act_prob)
            else:
                raise Exception("Uncaptured case")

        elif (var_type == 'action'):
            dist = np.array(1)
            if (case_type == "rwd"):
                for var in reversed(truth['row_var']):
                    if ('pr' in var and len(var) < 4):
                        prob = np.array([1. - self.approx_reward_dist[time+1][var], self.approx_reward_dist[time+1][var]])
                    else:
                        prob = np.array([1. - self.approx_state_dist[time][var], self.approx_state_dist[time][var]])
                    dist = np.outer(prob, dist).flatten()
            elif (case_type == "trans"):
                for var in reversed(truth['row_var']):
                    if (var[-1] == "'"):
                        var = var.replace("'", "")
                        prob = np.array([1. - self.approx_state_dist[time+1][var], self.approx_state_dist[time+1][var]])
                    else:
                        prob = np.array([1. - self.approx_state_dist[time][var], self.approx_state_dist[time][var]])
                    dist = np.outer(prob, dist).flatten()
            else:
                raise Exception("Uncaptured case")
            act_prob = np.ones(len(self.atomic_action_lst))
            approx_matrix  = np.outer(dist, act_prob)   

        else:
            if (case_type == 'same_time_cumu'):
                dist = np.array(1)
                for var in reversed(truth['row_var']):
                    if (var == 'r'):
                        var = var + str(time)
                    if (var != var_name):
                        if (var == 'pc0'):
                            prob = np.array([1., 1.])
                        else:
                            prob = np.array([1. - self.approx_reward_dist[time][var], self.approx_reward_dist[time][var]])
                    else:
                        prob = np.ones(2)
                    dist = np.outer(prob, dist).flatten()
                approx_matrix = dist

            elif (case_type == 'cross_time_cumu'):
                dist = np.array(1)
                for var in reversed(truth['row_var']):
                    if (var != var_name):
                        if ('c' in var):
                            if (var == 'c0'):
                                prob = np.array([1., 1.])
                            else:
                                prob = np.array([1. - self.approx_connect_dist[var], self.approx_connect_dist[var]])
                        else:
                            var_t = int(var[1])
                            prob = np.array([1. - self.approx_reward_dist[var_t][var], self.approx_reward_dist[var_t][var]])
                    else:
                        prob = np.ones(2)
                    dist = np.outer(prob, dist).flatten()
                approx_matrix = dist

            elif (case_type == 'rwd'):
                dist = np.array(1)
                for var in reversed(truth['row_var']):
                    if (var != var_name):
                        if ('pr' in var and var[2:].isnumeric()):
                            prob = np.array([1. - self.approx_reward_dist[time+1][var], self.approx_reward_dist[time+1][var]])
                        else:
                            prob = np.array([1. - self.approx_state_dist[time-1][var], self.approx_state_dist[time-1][var]])
                    else:
                        prob = np.ones(2)
                    dist = np.outer(prob, dist).flatten()

                if (var_type == 'rwd'):
                    act_prob = np.array(self.approx_action_dist[time-1])
                else:
                    act_prob = np.array(self.approx_action_dist[time])
                approx_matrix = np.outer(dist, act_prob)

            elif (case_type == 'trans'):
                dist = np.array(1)
                for var in reversed(truth['row_var']):
                    if (var != var_name):
                        if (var[-1] == "'"):
                            var = var.replace("'", "")
                            prob = np.array([1. - self.approx_state_dist[time+1][var], self.approx_state_dist[time+1][var]])
                        else:
                            prob = np.array([1. - self.approx_state_dist[time][var], self.approx_state_dist[time][var]])
                    else:
                        prob = np.ones(2)
                    dist = np.outer(prob, dist).flatten()

                if (var_type == 'rwd'):
                    act_prob = np.array(self.approx_action_dist[time-1])
                else:
                    act_prob = np.array(self.approx_action_dist[time])
                approx_matrix = np.outer(dist, act_prob)
            else:
                raise Exception("Uncaptured case")

        return approx_matrix

    def generate_masks(self, time, var_name, var_type, case_id, case_type):
        '''
        Generate matrix w.r.t the variable that is treated as the random variable.
        '''
        if (case_type == 'rwd'):
            truth = self.joint_rwd[case_id]
        elif (case_type == 'trans'):
            truth = self.joint_trans[case_id]
        elif (case_type == 'same_time_cumu'):
            truth = self.same_time_joint_cumu[case_id]
        elif (case_type == 'cross_time_cumu'):
            truth = self.cross_time_joint_cumu[case_id]
        else:
            raise Exception("Uncaptured case")

        if (var_type != 'action'):
            if (case_type == 'same_time_cumu'):
                dist0 = np.array(1)
                dist1 = np.array(1)
                for var in reversed(truth['row_var']):
                    if (var == 'r'):
                        var = var + str(time)
                    if (var == var_name):
                        mask0 = np.array([1, 0])
                        mask1 = np.array([0, 1])
                        dist0 = np.outer(mask0, dist0).flatten()
                        dist1 = np.outer(mask1, dist1).flatten()
                    else:
                        mask = np.array([1, 1])
                        dist0 = np.outer(mask, dist0).flatten()
                        dist1 = np.outer(mask, dist1).flatten()
                masks = [dist0, dist1]

            elif (case_type == 'cross_time_cumu'):
                dist0 = np.array(1)
                dist1 = np.array(1)
                for var in reversed(truth['row_var']):
                    if (var == var_name):
                        mask0 = np.array([1, 0])
                        mask1 = np.array([0, 1])
                        dist0 = np.outer(mask0, dist0).flatten()
                        dist1 = np.outer(mask1, dist1).flatten()
                    else:
                        mask = np.array([1, 1])
                        dist0 = np.outer(mask, dist0).flatten()
                        dist1 = np.outer(mask, dist1).flatten()
                masks = [dist0, dist1]

            else:
                dist0 = np.array(1)
                dist1 = np.array(1)
                for var in reversed(truth['row_var']):
                    if (var == var_name):
                        mask0 = np.array([1, 0])
                        mask1 = np.array([0, 1])
                        dist0 = np.outer(mask0, dist0).flatten()
                        dist1 = np.outer(mask1, dist1).flatten()
                    else:
                        mask = np.array([1, 1])
                        dist0 = np.outer(mask, dist0).flatten()
                        dist1 = np.outer(mask, dist1).flatten()
                act_prob = np.ones(len(self.atomic_action_lst
                ))
                masks = [np.outer(dist0, act_prob), np.outer(dist1, act_prob)]

        else:
            dist = np.ones(pow(2, len(truth['row_var'])))
            num_of_act = len(self.atomic_action_lst)
            act_prob = np.diag([1.]*num_of_act)
            masks = [np.outer(dist, act_prob[i]) for i in range(num_of_act)]

        return masks

    def update_approx(self, time, var_name, var_type, prob):
        '''
        Update approximate distribution.
        '''
        if (var_type == "state"):
            self.approx_state_dist[time][var_name] = prob[1]
        elif (var_type == "action"):
            self.approx_action_dist[time] = prob
        elif (var_type == 'rwd'):
            self.approx_reward_dist[time][var_name
            ] = prob[1]
        elif (var_type == 'cumu'):
            self.approx_connect_dist[var_name] = prob[1]
        else:
            raise Exception("Uncaptured case")

    def update_vec_state(self, time, var_name, rwd_type):
        '''
        Update state distribution in vectorized form.
        '''
        num_of_vals = 2
        log_p = [0.]*num_of_vals
        prob = [0.]*num_of_vals

        # the state variable as parent in transition
        if (var_name in self.trans_group.keys()):
            for case_id in self.trans_group[var_name]:
                approx = self.generate_approx(time, var_name, "state", case_id, "trans", rwd_type)
                truth = self.joint_trans[case_id]["table"]
                masks = self.generate_masks(time, var_name, "state", case_id, "trans")
                prev_log_p = copy.deepcopy(log_p)
                expec = np.multiply(approx, np.log(truth))

                p0 = np.multiply(expec, masks[0])
                p0 = p0[~np.all(p0 == 0, axis=1)]
                p1 = np.multiply(expec, masks[1])
                p1 = p1[~np.all(p1 == 0, axis=1)]
                indicator = 1. - np.equal(p0, p1)
                sharp_p = [np.multiply(p0, indicator), np.multiply(p1, indicator)]

                for idx in range(num_of_vals):
                    log_p[idx] += np.sum(sharp_p[idx])

        # the state variable in reward function
        if (var_name in self.rwd_group.keys()):
            for case_id in self.rwd_group[var_name]:
                approx = self.generate_approx(time, var_name, "state", case_id, "rwd", rwd_type)
                if (rwd_type == 'exp'):
                    truth = self.joint_exp_rwd[case_id]["table"]
                elif (rwd_type == 'std'):
                    truth = self.joint_rwd[case_id]["table"]
                else:
                    raise Exception("Uncaptured reward type")
                masks = self.generate_masks(time, var_name, "state", case_id, "rwd")

                expec = np.multiply(approx, np.log(truth))

                for idx in range(num_of_vals):
                    log_p[idx] += np.sum(np.multiply(expec, masks[idx]))

        child_var_name = var_name + "'"
        if (child_var_name in self.trans_group.keys()):
            for case_id in self.trans_group[child_var_name]:
                approx = self.generate_approx(time-1, child_var_name, "state", case_id, "trans", rwd_type)
                truth = self.joint_trans[case_id]["table"]
                masks = self.generate_masks(time, child_var_name, "state", case_id, "trans")

                expec = np.multiply(approx, np.log(truth))

                for idx in range(num_of_vals):
                    log_p[idx] += np.sum(np.multiply(expec, masks[idx]))

        norm_factor = sum(math.exp(log_p[i]) for i in range(num_of_vals))
        for idx in range(num_of_vals):
            prob[idx] = math.exp(log_p[idx])/norm_factor
        self.update_approx(time, var_name, 'state', prob)

    def update_vec_action(self, time, rwd_type):
        '''
        Update action distribution in vectorized form.
        '''
        num_of_vals = len(self.atomic_action_lst)
        log_p = [0.]*num_of_vals
        prob = [0.]*num_of_vals

        # the action variable in transition
        for case_id, truth in enumerate(self.joint_trans):
            approx = self.generate_approx(time, None, "action", case_id, "trans", rwd_type)
            masks = self.generate_masks(time, None, "action", case_id, "trans")
            truth = truth["table"]

            expec = np.multiply(approx, np.log(truth))
            for idx in range(num_of_vals):
                log_p[idx] += np.sum(np.multiply(expec, masks[idx]))

        # the action variable in reward function
        for case_id in self.rwd_group['action']:
            approx = self.generate_approx(time, None, "action", case_id, "rwd", rwd_type)
            masks = self.generate_masks(time, None, "action", case_id, "rwd")
            if (rwd_type == 'exp'):
                truth = self.joint_exp_rwd[case_id]["table"]
            elif (rwd_type == 'std'):
                truth = self.joint_rwd[case_id]["table"]
            else:
                raise Exception('Uncaptured case')

            expec = np.multiply(approx, np.log(truth))

            for idx in range(num_of_vals):
                log_p[idx] += np.sum(np.multiply(expec, masks[idx]))

        log_p += np.log(self.vec_policy(time))

        norm_factor = sum(math.exp(log_p[i]) for i in range(num_of_vals))
        for i in range(num_of_vals):
            prob[i] = math.exp(log_p[i])/norm_factor
        self.update_approx(time, None, 'action', prob)

    def update_vec_rwd(self, time, var_name, var_type, rwd_type):
        '''
        Update reward distribution in vectorized form.
        '''
        num_of_vals = 2
        log_p = [0.] * num_of_vals
        prob = [0.] * num_of_vals
        check = False

        if (var_type == 'final'):
            for case_id in self.cross_time_cumu_group[var_name]:
                approx = self.generate_approx(time, var_name, "rwd", case_id, "cross_time_cumu", rwd_type)
                truth = self.cross_time_joint_cumu[case_id]["table"]
                masks = self.generate_masks(time, var_name, "rwd", case_id, "cross_time_cumu")

                expec = np.multiply(approx, np.log(truth))

                for idx in range(num_of_vals):
                    log_p[idx] += np.sum(np.multiply(expec, masks[idx]))

        if (var_type == 'rwd'):
            for case_id in self.rwd_group[var_name]:
                approx = self.generate_approx(time, var_name, "rwd", case_id, "rwd", rwd_type)
                if (rwd_type == 'exp'):
                    truth = self.joint_exp_rwd[case_id]["table"]
                elif (rwd_type == 'std'):
                    truth = self.joint_rwd[case_id]["table"]
                else:
                    raise Exception("Uncaptured reward type")

                masks = self.generate_masks(time, var_name, "rwd", case_id, "rwd")
                
                expec = np.multiply(approx, np.log(truth))
             
                for idx in range(num_of_vals):
                    log_p[idx] += np.sum(np.multiply(expec, masks[idx]))

        if (var_name[0] == 'r'):
            sub_var = var_name[0]
        else:
            sub_var = var_name
        for case_id in self.same_time_cumu_group[sub_var]:
            approx = self.generate_approx(time, var_name, "rwd", case_id, "same_time_cumu", rwd_type)
            truth = self.same_time_joint_cumu[case_id]["table"]
            masks = self.generate_masks(time, var_name, "rwd", case_id, "same_time_cumu")

            expec = np.multiply(approx, np.log(truth))
            for idx in range(num_of_vals):
                log_p[idx] += np.sum(np.multiply(expec, masks[idx]))

        norm_factor = sum(math.exp(log_p[i]) for i in range(num_of_vals))
        for idx in range(num_of_vals):
            prob[idx] = math.exp(log_p[idx])/norm_factor

        self.update_approx(time, var_name, 'rwd', prob)

    def update_vec_cumu(self, time, var_name, rwd_type):
        '''
        Update cumulative reward distribution in vectorized form.
        '''
        num_of_vals = 2
        log_p = [0.] * num_of_vals
        prob = [0.] * num_of_vals

        # the reward variable as parent variable
        for case_id in self.cross_time_cumu_group[var_name]:
            approx = self.generate_approx(time, var_name, 'cumu', case_id, 'cross_time_cumu', rwd_type)
            truth = self.cross_time_joint_cumu[case_id]["table"]
            masks = self.generate_masks(time, var_name, 'cumu', case_id, 'cross_time_cumu')

            expec = np.multiply(approx, np.log(truth))

            for idx in range(num_of_vals):
                log_p[idx] += np.sum(np.multiply(expec, masks[idx]))

        norm_factor = sum(math.exp(log_p[i]) for i in range(num_of_vals))
        for idx in range(num_of_vals):
            prob[idx] = math.exp(log_p[idx])/norm_factor
        self.update_approx(time, var_name, 'cumu', prob)

    def _pre_a_seq_sampling(self, size):
        '''
        Sampling action sequence based on approximate action distribution.
        '''
        a_seq = np.empty((self.h, size), dtype=str)
        for t in range(0, self.h):
            a_dist = self.approx_action_dist[t]
            a_seq[t] = np.array(random.choices(list(range(self.num_of_action)), a_dist, k=size))
        a_seq = a_seq.astype(int)
        return a_seq 

    def _update_a_seq(self, a_seq, time, action, size):
        '''
        Update action sequence w.r.t the action value to be considered.
        '''
        a_seq[time] = np.array([action]*size)
        return a_seq.T
    
    def approx_a_update(self, time, rwd_type):
        '''
        Update approximate action distribution.
        '''
        # np array of initial state
        init_state_lst = list(map(int, self.init_state.values()))
        init_sarray = np.array(init_state_lst)

        init_states = np.tile(init_sarray, (self.full_sample_size, 1))
        # initial state matrix with each column to be one initial state, number of column is the number of samples
        init_states = init_states.T
        size = self.a_seq_size

        q_update = [0.]*self.num_of_action

        pre_a_seqs = self._pre_a_seq_sampling(size)
        for idx, act in enumerate(self.atomic_action_lst):
            policy_part = math.log(float(self.policy[time][act]))
            # a matrix of each row to be one action sequence, the number of rows is the number of action samples

            a_seqs = self._update_a_seq(pre_a_seqs, time, idx, size)
            q_update[idx] = self._log_likelihood_estimate(init_states, a_seqs, size, rwd_type) + policy_part

        denominator = 0
        for q in q_update:
            denominator += math.exp(q)
        if (denominator == 0):
            for idx in range(self.num_of_action):
                self.approx_action_dist[time][idx] = 1./self.num_of_action
        else:
            for idx in range(self.num_of_action):
                self.approx_action_dist[time][idx] = math.exp(q_update[idx])/denominator

    def _log_likelihood_estimate(self, init_states, a_seqs, a_size, rwd_type):
        '''
        Estimate the log likelihood of the total reward to be optimal.
        '''
        size = self.full_traj_sample_size
        # a matrix of each row to be one action sequence, the number of rows are the total number of samples        
        extended_a_seqs = np.tile(a_seqs, (size, 1))
        # an binary array recording the cumulative reward results for each trajectory

        c_array = self._traj_sampling(init_states, extended_a_seqs, rwd_type)
        c_array = c_array.reshape((size, a_size))

        c_count_per_a = np.sum(c_array, 0)
        log_est = np.log((c_count_per_a+0.01)/(size+0.02))
        if (len(log_est) != a_size):
            raise Exception("Matrix dimension error")
        return np.sum(log_est)/a_size

    def _extract_state_grounding(s_dict, s_vars):
        return ''.join(s_dict[s] for s in s_vars)

    def _extract_trans_dist(self, curr_states, a_idxs):
        '''
        vectorized transition extraction for all state variables
        '''
        a_array = np.zeros((self.num_of_action, self.full_sample_size)) 
        # a matrix of each column to be one action sample of a certain time step
        a_array[a_idxs, np.arange(self.full_sample_size)] = 1
        probs = []

        # a matrix of probability of child var to be 1, each column to be one sample, each row corresponds to one state variable

        temp_s = np.dot(self.trans_factor_matrix, curr_states)
        s1_idx_array = np.dot(self.trans_idx_matrix, temp_s) + 1
        s1_idx_array = s1_idx_array.astype(int)

        for idx1, trans in zip(s1_idx_array, self.joint_trans):

            s_array = np.zeros((len(trans['table']), self.full_sample_size))
            s_array[idx1, np.arange(self.full_sample_size)] = 1

            # one hot state vector * transition matrix * one hot action vector = trans probability in matrix form to parallel samples
            temp_p = np.dot(trans['table'], a_array)
            prob = np.sum(np.multiply(s_array, temp_p), 0)
            probs.append(prob)
        p1_array = np.array(probs)
        return p1_array

    def _vec_soften(self, ary):
        '''
        Parameter need to be the same as in spudd_parser.py
        '''
        eps = 1e-4
        ary[np.where(ary >= 1.- eps)] = 1. - eps
        ary[np.where(ary <= eps)] = eps
        return ary

    def _extract_mega_rwd_dist(self, curr_states, a_idxs):
        '''
        vectorized reward for CSVI
        '''
        a_array = np.zeros((self.num_of_action, self.full_sample_size)) 
        a_array[a_idxs, np.arange(self.full_sample_size)] = 1
        raws = []

        # a matrix of #rwd cases * #samples
        temp_r = np.dot(self.rwd_factor_matrix, curr_states)
        r1_idx_array = np.dot(self.rwd_idx_matrix, temp_r) + 1
        r1_idx_array = r1_idx_array.astype(int)

        for idx1, rwds in zip(r1_idx_array, self.joint_raw_rwd):
            r_array = np.zeros((len(rwds['table']), self.full_sample_size))
            r_array[idx1, np.arange(self.full_sample_size)] = 1
            # one hot state vector * reward matrix * one hot action vector = trans probability in matrix form to parallel samples

            temp_rp = np.dot(rwds['table'], a_array)
            raw = np.sum(np.multiply(r_array, temp_rp), 0)

            raws.append(raw)

        p1_array = np.array(raws)
        unf_r = np.sum(p1_array, 0)
        unf_p_array = (unf_r - (self.r_min_unf - self.epsilon))/((self.r_max_unf - (self.r_min_unf - self.epsilon)))
        p_array = self._vec_soften(unf_p_array)
        
        return p_array
 
    def _extract_mega_exp_rwd_dist(self, curr_states, a_idxs):
        '''
        vectorized reward for exp CSVI
        '''
        a_array = np.zeros((self.num_of_action, self.full_sample_size)) 
        a_array[a_idxs, np.arange(self.full_sample_size)] = 1
        raws = []

        # a matrix of #rwd cases * #samples
        temp_r = np.dot(self.rwd_factor_matrix, curr_states)
        r1_idx_array = np.dot(self.rwd_idx_matrix, temp_r) + 1
        r1_idx_array = r1_idx_array.astype(int)

        for idx1, rwds in zip(r1_idx_array, self.joint_raw_rwd):
            r_array = np.zeros((len(rwds['table']), self.full_sample_size))
            r_array[idx1, np.arange(self.full_sample_size)] = 1
            # one hot state vector * reward matrix * one hot action vector = trans probability in matrix form to parallel samples

            temp_rp = np.dot(rwds['table'], a_array)
            raw = np.sum(np.multiply(r_array, temp_rp), 0)

            raws.append(raw)

        p1_array = np.array(raws)
        unf_r = np.sum(p1_array, 0)
        unf_p_array = np.exp(unf_r)/np.exp(self.r_max_unf)
        p_array = self._vec_soften(unf_p_array)
        
        return p_array
        
    def _traj_sampling(self, init_states, a_seqs, rwd_type):
        '''
        Given action sequence, sample the trajectory w.r.t. state and reward.
        - init_states: a matrix of each column to be one initial state, #column = total samples
        - a_seq: a matrix of each row to be one action sequence, #rows = total samples
        '''
        curr_states = init_states
        c_var = np.ones(self.full_sample_size)
        for t in range(0, self.h):
            state_probs = self._extract_trans_dist(curr_states, a_seqs[:, t])
            next_states = np.random.binomial(1, state_probs)
            next_states = next_states.astype(int)
            if (rwd_type == 'std'):
                rvar_probs = self._extract_mega_rwd_dist(curr_states, a_seqs[:, t])
            elif (rwd_type == 'exp'):
                rvar_probs = self._extract_mega_exp_rwd_dist(curr_states, a_seqs[:, t])
            else:
                raise Exception('reward type parameter not supported, only support std and exp')
            r_var = np.random.binomial(1, rvar_probs)

            c_var = c_var.astype(int)
            r_var = r_var.astype(int)
            c1_cond = c_var*4 + r_var*2 + 1

            c1_matrix = np.zeros((8, self.full_sample_size))
            c1_matrix[c1_cond, np.arange(self.full_sample_size)] = 1
            c_dist = np.dot(self.cross_time_joint_cumu[t]['table'], c1_matrix)
            c_var = np.random.binomial(1, c_dist)

            curr_states = copy.deepcopy(next_states)
        return c_var

    def _a_seq_sampling(self, time, action, size):
        '''
        Sampling action sequence based on approximate action distribution.
        '''
        a_seq = np.empty((self.h, size), dtype=str)
        if (time != -1):
            a_seq[time] = np.array([action]*size)
        for t in range(0, self.h):
            if (t != time):
                a_dist = self.approx_action_dist[t]
                a_seq[t] = np.array(random.choices(list(range(self.num_of_action)), a_dist, k=size))
        a_seq = a_seq.astype(int)
        return a_seq.T

    def calc_elbo(self, alg, rwd_type):
        '''
        ELBO calculation of MFVI and CSVI for analysis purpose.
        '''
        if ('mfvi' in alg):
            # policy part
            part1 = 0.
            for time in range(0, self.h):
                part1 += np.inner(np.array(self.approx_action_dist[time]), np.log(self.vec_policy(time)))

            # transition part
            part2 = 0.
            for time in range(1, self.h):
                for case_id, case in enumerate(self.joint_trans):
                    truth = case['table']
                    approx = self.generate_approx(time, None, "elbo", case_id, "trans", rwd_type)
                    part2 += np.sum(np.multiply(np.log(truth), approx))

            # reward part
            part3 = 0.
            rwd = None
            for time in range(1, self.h+1):
                if (rwd_type == "std"):
                    rwd = self.joint_rwd
                elif (rwd_type == "exp"):
                    rwd = self.joint_exp_rwd
                else:
                    raise Exception("Uncaptured case for reward elbo calculation")
                for case_id, case in enumerate(rwd):
                    truth = case['table']
                    approx = self.generate_approx(time, None, 'elbo', case_id, "rwd", rwd_type)
                    part3 += np.sum(np.multiply(np.log(truth), approx))

                for case_id, case in enumerate(self.same_time_joint_cumu):
                    truth = case['table']
                    approx = self.generate_approx(time, None, 'elbo', case_id, "same_time_cumu", rwd_type)
                    part3 += np.sum(np.multiply(np.log(truth), approx))
                    
            # connection part
            part4 = 0.
            for case_id, case in enumerate(self.cross_time_joint_cumu):
                truth = case['table']
                approx = self.generate_approx(None, None, 'elbo', case_id, "cross_time_cumu", rwd_type)
                part4 += np.sum(np.multiply(np.log(truth), approx))

            # approx state part
            part5 = 0.
            for time in range(1, self.h):
                for s_var in self.state_vars:
                    part5 += math.log(self.approx_state_dist[time][s_var]) * self.approx_state_dist[time][s_var]
                    part5 += math.log(1. - self.approx_state_dist[time][s_var]) * (1. - self.approx_state_dist[time][s_var])

            # approx action part
            part6 = 0.
            for time in range(0, self.h):
                part6 += np.inner(np.array(self.approx_action_dist[time]), np.log(np.array(self.approx_action_dist[time])))
                part6 += np.inner(1. - np.array(self.approx_action_dist[time]), np.log(1. - np.array(self.approx_action_dist[time])))

            # approx reward part
            part7 = 0.
            for time in range(1, self.h+1):
                for var in self.approx_reward_dist[time].keys():
                    part7 += math.log(self.approx_reward_dist[time][var]) * self.approx_reward_dist[time][var]
                part7 += math.log(1. - self.approx_reward_dist[time][var]) * (1. - self.approx_reward_dist[time][var])

            # approx connect part
            part8 = 0.
            if (self.h > 1):
                for time in range(1, self.h):
                    part8 += math.log(self.approx_connect_dist['c'+str(time)]) * self.approx_connect_dist['c'+str(time)]
                    part8 += math.log(1. - self.approx_connect_dist['c'+str(time)]) * (1. - self.approx_connect_dist['c'+str(time)])

            elbo = part1 + part2 + part3 + part4 - part5 - part6 - part7 - part8
        else:
            # policy part
            policy_part = 0.
            for time in range(0, self.h):
                policy_part += np.inner(np.array(self.approx_action_dist[time]), np.log(self.vec_policy(time)))

            traj_part = 0.
            a_seqs = self._a_seq_sampling(-1, None, self.a_seq_size)
            init_state_lst = list(map(int, self.init_state.values()))
            init_sarray = np.array(init_state_lst)
            init_states = np.tile(init_sarray, (self.full_sample_size, 1))
            init_states = init_states.T
            traj_part += self._log_likelihood_estimate(init_states, a_seqs, self.a_seq_size, rwd_type)

            approx_part = 0.
            for time in range(0, self.h):
                approx_part += np.inner(np.array(self.approx_action_dist[time]), np.log(np.array(self.approx_action_dist[time])))
                approx_part += np.inner(1. - np.array(self.approx_action_dist[time]), np.log(1. - np.array(self.approx_action_dist[time])))
            elbo = policy_part + traj_part - approx_part
        return elbo

    def get_init(self):
        return self.init_state

if __name__ == "__main__":
    file = '../spudd_sperseus_test2/skill_teaching_inst_mdp__1.spudd'
    parsed_ins = SPUDD_Parser(file)
    p = configargparse.ArgParser(default_config_files=['../conf/default.conf'])
    p.add('-i', '--instance', nargs='+', help='full spudd file name of the instance')
    p.add('-a', '--algorithm', nargs='+', help='algorithm name shortcut, including: mfvi_bwd, mfvi_fwd, csvi_bwd, csvi_fwd, bwd_lbp, fwd_lbp, exp_mfvi_bwd, exp_csvi_bwd, mfvi_bwd_noS, demo_mfvi_bwd, demo_mfvi_bwd_noS, demo_mfvi_bwd_med, demo_mfvi_bwd_main_all, demo_csvi_bwd, random.')
    p.add('-s', '--simu_type', help=('simulation type, only support cost'))
    p.add('-app', '--alg_param_path', help=('relative path to algorithm parameters configuration'))
    p.add('-vpp', '--vis_param_path', help=('relative path to visulization parameters configuration'))

    args = p.parse_args()
    simu = args.simu_type
    pp = args.alg_param_path

    search_horizon = 5

    var_inf = Variational_inference(None, parsed_ins, search_horizon, pp)
    init_state = parsed_ins.get_init_state()
    ATOMIC_ACTIONS = parsed_ins.get_atomic_action_lst()
    STATE_VARIABLES = parsed_ins.get_state_vars()
    uniform_policy = init_factored_policy(ATOMIC_ACTIONS, search_horizon, uniform=True)
    var_inf.init_approx(init_state, uniform_policy, search_horizon)

    num_case = var_inf.get_num_of_rwd_case()
    print(num_case)
    print(var_inf.get_reward_approx_dist(2, 'r'))
    print(var_inf.get_connect_approx_dist('c'+str(search_horizon-1)))
    print(var_inf.get_action_approx_dist(2))
    print(var_inf.get_state_approx_dist(1, STATE_VARIABLES[1]))

    for _ in range(10):
        for t in range(search_horizon):
            if (t > 0):
                for svar in STATE_VARIABLES:
                    var_inf.update_vec_state(t, svar)

            var_inf.update_vec_action(t)

            for r_idx in range(num_case):
                var_inf.update_vec_rwd(t+1, 'pr'+str(r_idx+1), 'rwd')
                if (r_idx == num_case - 1):
                    var_inf.update_vec_rwd(t+1, 'r', 'final')
                else:
                    var_inf.update_vec_rwd(t+1, 'pc'+str(r_idx+1), 'cumu')

            if (t < search_horizon - 1):
                var_inf.update_vec_cumu(t+1, 'c'+str(t+1))

    print("update reward")
    for t in range(search_horizon):
        print(var_inf.get_reward_approx_dist(t+1, 'r'))
    print("\n")
    print("update connect")
    for t in range(search_horizon):
        print(var_inf.get_connect_approx_dist('c'+str(t+1)))
    print("\n")
    print("update action")
    for t in range(search_horizon):
        print(var_inf.get_action_approx_dist(t))
    print("\n")
    print("update state")
    for t in range(search_horizon):
        for svar in STATE_VARIABLES:
            print(var_inf.get_state_approx_dist(t, svar))