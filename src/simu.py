import time as timing
import numpy as np
import multiprocessing as mp

import math
import random
import json
import os
from tqdm import tqdm

from demo_vi import demo_mfvi_bwd_main, demo_csvi_bwd_main
from csvi import csvi_bwd_main, csvi_fwd_main
from lbp import lbp_bwd_main, lbp_fwd_main, lbp_fwd_main_jax
from mfvi import mfvi_bwd_main,  mfvi_fwd_main
from mfvi_noS import mfvi_bwd_noS_main, mfvi_fwd_noS_main
from variational_inf import Variational_inference

from utils import reduce_to_factor_jax, record


class Evaluation:

    def __init__(self, simu_type, parser, initial_policy, look_ahead, log_file, ins_name, param_path):

        # Representation of state variables
        self.s_vars = parser.get_state_vars()
        # Representation of action variables
        self.a_vars = parser.get_action_vars()
        # Representation of action space
        self.valid_actions = parser.get_valid_action_lst()
        self.atomic_actions = parser.get_atomic_action_lst()
        # Representation of transition dynamics
        self.trans = parser.get_trans()
        # Horizon
        self.T = parser.get_horizon()
        # Discount
        self.gamma = parser.get_discount()

        # Representation of reward distribution (normailized reward table)
        self.reward_table = parser.get_reward_table()

        self.uniform_policy = initial_policy
        self.initial_policy = initial_policy
        # Initial state
        self.init_state = parser.get_init_state()
        self.rdm = np.random.RandomState()
        self.look_ahead_length = look_ahead

        self.parent_s_table = parser.get_parent_s_table()
        self.parent_a_table = parser.get_parent_a_table()
        self.pseudo_trans = parser.get_pseudo_trans()

        self.log_path = log_file
        self.simu_type = simu_type
        self.ins_name = ins_name
        self.timing = {}
        self.param_path = param_path

    def _accumulate_rwd(self, state, action):
        '''
        Collect reward in simulation.
        '''
        state_dict = {}
        for svar in self.s_vars:
            if (state[svar] == '1'):
                state_dict[svar] = 'true'
            else:
                state_dict[svar] = 'false'

        rwd = self.reward_table[action]['extra']
        for case in self.reward_table[action]['cases']:
            parent = case
            finished_search = False
            updated = False
            maxiter = len(state)
            count = 0
            while (not finished_search and count <= maxiter):
                for key_p in parent.keys():
                    for key_c in parent[key_p].keys():
                        if (key_c == state_dict[key_p]):
                            parent = parent[key_p][key_c]
                            count += 1
                            updated = True
                            if (not type(parent) is dict):
                                rwd += float(parent)
                                finished_search = True
                            break
                    if (updated):
                        updated = False
                    else:
                        finished_search = True

            if (count > maxiter):
                raise ValueError("reward tree is deeper than the number of state variables")
        return rwd

    def _find_best_action(self, dist):
        '''
        From action distribution to get the concrete action.
        '''
        max_val = 0
        duplicate = False
        action = None
        if (isinstance(dist, dict)):
            for key in dist.keys():
                if (dist[key] > max_val):
                    max_val = dist[key]
                    action = key
                    duplicate = False
                elif (dist[key] == max_val):
                    duplicate = True
            if (duplicate):
                candid = []
                for key in dist.keys():
                    if (dist[key] == max_val):
                        candid.append(key)
                action = self.rdm.choice(np.array(candid))
        else:
            max_val = max(dist)
            candid = []
            for idx, val in enumerate(dist):
                if (val == max_val):
                    candid.append(self.atomic_actions[idx])
            action = self.rdm.choice(np.array(candid))
        return action

    def _policy_calc(self, alg, parser, alg_struc, curr_state, remain_con, remain_policy, search_horizon, log_path, *args):
        '''
        Calculate the action distribution for different algorithms.
        '''
        with open(self.param_path, 'r') as params_file:
            params = json.load(params_file)
            mfvi_max_iter = params['mfvi_max_iter']
            csvi_max_iter = params['csvi_max_iter']
            conv_threshold = params['vi_conv_threshold']
            fwd_lbp_iter = params['fwd_lbp_iter_num']
            bwd_lbp_iter = params['bwd_lbp_iter_num']

        if (alg == 'mfvi_bwd'):
            return mfvi_bwd_main(parser, alg_struc, search_horizon, log_path, mfvi_max_iter, conv_threshold, rwd_type='std')
        elif (alg == 'mfvi_bwd_noS'):
            return mfvi_bwd_noS_main(parser, alg_struc, search_horizon, log_path, mfvi_max_iter, conv_threshold)
        elif (alg == 'exp_mfvi_bwd'):
            return mfvi_bwd_main(parser, alg_struc, search_horizon, log_path, mfvi_max_iter, conv_threshold, rwd_type='exp')
        elif (alg == 'csvi_bwd'):
            return csvi_bwd_main(parser, alg_struc, search_horizon, log_path, csvi_max_iter, conv_threshold, rwd_type='std')
        elif (alg == 'exp_csvi_bwd'):
            return csvi_bwd_main(parser, alg_struc, search_horizon, log_path, csvi_max_iter, conv_threshold, rwd_type='exp')

        elif (alg == 'mfvi_fwd'):
            em_step = args[0]
            return mfvi_fwd_main(parser, alg_struc, search_horizon, em_step, log_path, mfvi_max_iter, conv_threshold, rwd_type='std')
        elif (alg == 'mfvi_fwd_noS'):
            em_step = args[0]
            return mfvi_fwd_noS_main(parser, alg_struc, search_horizon, em_step, log_path, mfvi_max_iter, conv_threshold)
        elif (alg == 'exp_mfvi_fwd'):
            em_step = args[0]
            return mfvi_fwd_main(parser, alg_struc, search_horizon, em_step, log_path, mfvi_max_iter, conv_threshold, rwd_type='exp')
        elif (alg == 'csvi_fwd'):
            em_step = args[0]
            return csvi_fwd_main(parser, alg_struc,  search_horizon, em_step, log_path, csvi_max_iter, conv_threshold, rwd_type='std')
        elif (alg == 'exp_csvi_fwd'):
            em_step = args[0]
            return csvi_fwd_main(parser, alg_struc, search_horizon, em_step, log_path, csvi_max_iter, conv_threshold, rwd_type='exp')
        elif (alg == 'bwd_lbp'):
            return lbp_bwd_main(parser, alg_struc, curr_state, bwd_lbp_iter)
        elif (alg == 'fwd_lbp'):
            return lbp_fwd_main(parser, remain_policy, remain_con, search_horizon, curr_state)
        elif (alg == 'fwd_lbp_jax'):
            return lbp_fwd_main_jax(parser, alg_struc, remain_policy, search_horizon, curr_state, fwd_lbp_iter)

        elif (alg == 'demo_mfvi_bwd'):
            version = 'all'
            elbo_check = args[0]
            time = args[1]
            return demo_mfvi_bwd_main(parser, alg_struc, search_horizon, log_path, elbo_check, time, version, self.ins_name, mfvi_max_iter, conv_threshold)
        elif (alg == 'demo_mfvi_bwd_noS'):
            version = 'none'
            elbo_check = args[0]
            time = args[1]
            return demo_mfvi_bwd_main(parser, alg_struc, search_horizon, log_path, elbo_check, time, version, self.ins_name, mfvi_max_iter, conv_threshold)
        elif (alg == 'demo_mfvi_bwd_med'):
            version = 'med'
            elbo_check = args[1]
            time = args[2]
            return demo_mfvi_bwd_main(parser, alg_struc, search_horizon, log_path, elbo_check, time, version, self.ins_name, mfvi_max_iter, conv_threshold)
        elif (alg == 'demo_mfvi_bwd_main_all'):
            version = 'main_all'
            elbo_check = args[1]
            time = args[2]
            return demo_mfvi_bwd_main(parser, alg_struc, search_horizon, log_path, elbo_check, time, version, self.ins_name, mfvi_max_iter, conv_threshold)
        elif (alg == 'demo_csvi_bwd'):
            elbo_check = args[1]
            time = args[2]
            return demo_csvi_bwd_main(parser, alg_struc, search_horizon, log_path, elbo_check, time, self.ins_name, csvi_max_iter, conv_threshold)
        elif (alg == 'random'):
            return self.uniform_policy
        else:
            raise Exception("Algorithm parameter {} not found.".format(alg))

    def _evaluate(self, seed_base, seed, parser, log_path, alg, *args):
        '''
        Simulation of different algorithms of the full trajectories.
        '''
        connect_dist = args[0]
        # Current State
        curr_state = {}
        for s in self.init_state.keys():
            prob = self.init_state[s]
            curr_state[s] = str(self.rdm.choice(np.array([0, 1]), p=prob))
        # Cumulative Reward
        cumu_rwd = 0
        cumu_rwd_lst = []
        self.timing[alg] = []
        em_step = args[1]
        elbo_check = False
        if ('vi' in alg):
            alg_struc = Variational_inference(parser, self.look_ahead_length, self.param_path)
        elif ('bp' in alg):
            s_vars = parser.get_state_vars()
            a_vars = parser.get_action_vars()
            valid_actions = parser.get_valid_action_lst()
            trans_prob = parser.get_pseudo_trans()
            state_dependency = parser.get_state_dependency()
            normal_factor = parser.get_normal_factor()
            reward_dist = parser.get_reward_table()
            atomic_action_lst = parser.get_atomic_action_lst()
            if ('bwd' in alg):
                mes_dirc = 'bw'
            else:
                mes_dirc = 'fw'
            alg_struc = reduce_to_factor_jax(state_dependency, valid_actions, atomic_action_lst, reward_dist, connect_dist, trans_prob, self.initial_policy, s_vars, a_vars, self.look_ahead_length, mes_dirc , normal_factor, False)


        for time in tqdm(range(0, self.T), desc = '{}-inst{} simulation {}'.format(self.ins_name.split('_')[0].capitalize(), self.ins_name.split('_')[-1].split('.')[0], int(seed/seed_base) + 1), position=int(seed/seed_base)):

            if (alg.split('_')[0] == 'demo'):
                if (time >= 2 and time <= 10):
                    elbo_check = True
            record("At time {}, state {}\n".format(time, curr_state), log_path)
            search_horizon = min(self.look_ahead_length, self.T - time)
            remain_con = {key: connect_dist[key] for key in range(0, search_horizon + 1)}
            remain_policy = {key: self.initial_policy[key] for key in range(0, search_horizon)}
            if ('vi' in alg):
                alg_struc.init_approx(curr_state, remain_policy, search_horizon)
            
            elif ('lbp' in alg):
                # code could be optimized here
                if (search_horizon != self.look_ahead_length):
                    alg_struc = reduce_to_factor_jax(state_dependency, valid_actions, atomic_action_lst, reward_dist, remain_con, trans_prob, remain_policy, s_vars, a_vars, search_horizon, mes_dirc , normal_factor, False)   
            else:
                alg_struc = None

            start = timing.time()              
            if (alg == 'fwd_lbp'):
                action = self._policy_calc(alg, parser, alg_struc, curr_state, remain_con, remain_policy, search_horizon, log_path, em_step, elbo_check, time)
                end = timing.time()
                # record each step algorithm running time
                self.timing[alg].append(end - start)
                exe_start = timing.time()
                atomic_action = self.atomic_actions[self.valid_actions.index(action)]
            else:
                q_action_var = self._policy_calc(alg, parser, alg_struc, curr_state, remain_con, remain_policy, search_horizon, log_path, em_step, elbo_check, time)
                end = timing.time()
                # record each step algorithm running time
                self.timing[alg].append(end - start)

                # extract action from approximate distribution
                exe_start = timing.time()
                if (alg == 'bwd_lbp'):
                    best_prob = max(q_action_var[0])
                    candid_idx = []
                    for q_idx, prob in enumerate(q_action_var[0]):
                        if (prob == best_prob):
                            candid_idx.append(q_idx)
                    action_idx = random.choices(candid_idx)[0]
                    atomic_action = self.atomic_actions[action_idx]
                else:
                    atomic_action = self._find_best_action(q_action_var[0])
                record("At time {}, action distribution is \n{}\n, {} choose action {}\n".format(time, q_action_var[0], alg, atomic_action), log_path)
            next_state = {}

            for svar in self.s_vars:
                child_s_var = svar + "'"
                prob = []
                value = [0, 1]
                parent = self.trans[atomic_action][svar]
                is_leaf = False
                while (not is_leaf):
                    for key in parent.keys():
                        if (key in self.s_vars):
                            if (curr_state[key] == '1'):
                                parent = parent[key]['true']
                            else:
                                parent = parent[key]['false']
                        elif (key == child_s_var):
                            prob = [float(parent[key]['false'][-1]), float(parent[key]['true'][-1])]
                            is_leaf = True
                # extract next state variable value and corresponding probability
                next_state[svar] = str(self.rdm.choice(np.array(value), p=prob))
            exe_end = timing.time()
            rwd_start = timing.time()
            if (self.simu_type == 'reward'):
                cumu_rwd += self._accumulate_rwd(curr_state, atomic_action)
                cumu_rwd_lst.append(cumu_rwd)
            else:
                cumu_rwd -= self._accumulate_rwd(curr_state, atomic_action)
                cumu_rwd_lst.append(cumu_rwd)
            rwd_end = timing.time()
            curr_state = next_state

            record("{} algorithm taking step {} in horizon {}, policy calculation use {} second, execution use {} second, reward collection use {} second\n".format(alg.capitalize(), time, self.T, end - start, exe_end - exe_start, rwd_end - rwd_start), log_path)
        record("Cumulative reward of {} in this simulation is {}".format(alg, cumu_rwd), log_path)
        return cumu_rwd, cumu_rwd_lst

    def _run(self, queue, epis, seed_base, seed, parser, alg, *args):
        '''
        Record the reward of the simulation.
        '''
        start = timing.time()
        domain = parser.get_domain()
        rwd = []
        rwd_lst = []
        self.rdm.seed(seed)
        log_path = self.log_path + str(seed) + '_log.txt'
        for epi in range(0, epis):
            record("Running {}th simulation of algorithm {} in domain {} with {} state variables, {} action variables and horizon {}".format(epi, alg, domain, len(self.s_vars), len(self.a_vars), self.T), path=log_path)
            simu_rwd, simu_rwd_lst = self._evaluate(seed_base, seed, parser, log_path, alg, *args)
            record("Running {}th simulation of algorithm {} in domain {} has running time {}".format(epi, alg, domain, self.timing[alg]), path=log_path)
            rwd.append(simu_rwd)
            rwd_lst.append(simu_rwd_lst)
        end = timing.time()
        queue.put([rwd, rwd_lst])

    def simulation(self, parallel, intended_epis, parser, alg, *args):
        '''
        Set up parallel running of the simulation.
        '''
        if (parallel):
            partition = min(mp.cpu_count(), intended_epis)
        else:
            partition = 1

        domain = parser.get_domain().split('/')[-1]
        T = int(parser.get_horizon())

        seed_base=37

        processes = []

        que = mp.Queue()
        rwds = []
        record('Simulation use {} CPUs'.format(partition))

        # directory for storing rwd w.r.t time step information
        alg_part = alg.split('_')
        extra_info_dir = '../results/demo/temp/{}'.format(alg)
        if (not os.path.isdir(extra_info_dir)):
            os.makedirs(extra_info_dir, exist_ok=True)
        extra_info = extra_info_dir + '/{}_{}_temporal_elbo.json'.format(domain[:-6], alg)

        epis = math.ceil(intended_epis/partition)
        full_epis = epis*partition
        seeds = list(range(0, seed_base*partition, seed_base))
        start = timing.time()

        for idx in range(0, partition):
            p = mp.Process(target=self._run, args=(que, epis, seed_base, seeds[idx], parser, alg, *args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        while (not que.empty()):
            rwds.append(que.get())

        if (len(rwds) != partition):
            raise Exception('Not all simulation get their results, result {}, partition {}'.format(len(rwds), partition))
        end = timing.time()

        total_rwd = []
        temporal_rwd = [0]*T
        for rwd in rwds:
            for item, item_lst in zip(rwd[0], rwd[1]):
                total_rwd.append(item)
                if (alg_part[0] == 'demo'):
                    for t_idx, val in enumerate(item_lst):
                        temporal_rwd[t_idx] = temporal_rwd[t_idx] + val

        mean = sum(total_rwd)/full_epis
        variance = sum([((x - mean) ** 2) for x in total_rwd]) / max(full_epis*(full_epis - 1), 1)

        if (alg_part[0] == 'demo'):
            for idx, _ in enumerate(temporal_rwd):
                temporal_rwd[idx] = temporal_rwd[idx]/full_epis
            result = {'alg': alg, self.simu_type: temporal_rwd}
            with open(extra_info, 'w') as output:
                json.dump(result, output)

        std = variance ** 0.5
        return mean, std