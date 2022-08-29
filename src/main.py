import json
import os
from datetime import datetime
import sys

import configargparse

from utils import init_factored_policy, generate_connect_distribution
from simu import Evaluation
from spudd_parser import SPUDD_Parser


def generate_rlt(alg_choice, simu_type):
    '''
    Generate files for storing the results
    '''
    result = {}
    result['info'] = {'horizon': [], 's_num': [], 'a_num': [], 'simu_type': simu_type}
    result[alg_choice] = {'mean':[], 'std': []}
    return result


def update_rlt(alg_choice, horizon, s_num, a_num, mean, std, output_log, simu_type):
    '''
    Write results to files
    '''
    if (not os.path.exists(output_log)):
        init_result = generate_rlt(alg_choice, simu_type)
        with open(output_log, 'w') as output:
            json.dump(init_result, output)
    with open(output_log, 'r+') as json_file:
        result = json.load(json_file)
        result['info']['horizon'].append(horizon)
        result['info']['s_num'].append(s_num)
        result['info']['a_num'].append(a_num)
        result[alg_choice]['mean'].append(mean)
        result[alg_choice]['std'].append(std)
        json_file.seek(0)
        json.dump(result, json_file)


def run(simu_type, alg_choice, ins, ins_name, log_file, epids, em_steps, look_ahead_choices, output_log, param_path, parallel = True):
    '''
    Running experiments
    '''
    parsed_ins = SPUDD_Parser(ins)
    size_info = parsed_ins.get_problem_size()
    horizon = parsed_ins.get_horizon()
    STATE_VARIABLES = parsed_ins.get_state_vars()
    ACTION_VARIABLES = parsed_ins.get_action_vars()
    ATOMIC_ACTIONS = parsed_ins.get_atomic_action_lst()
    s_num = len(STATE_VARIABLES)
    a_num = len(ACTION_VARIABLES)
    print("Experiment of algorithm {} for instance {} with horizon {},\
{} state variable and {} action variable. State dependency maximum {},\
minimum {}, average {}. Valid action {}. Enumeration maximum {}, minimum {},\
average {}".format(alg_choice, ins, horizon, s_num, a_num, size_info[0],\
size_info[1], size_info[2], size_info[3], size_info[4], size_info[5], size_info[6]))

    for look_ahead in look_ahead_choices:
        ###-------------------------------------------
        # Simulation
        ### ------------------------------------------
        # Policy
        uniform_policy = init_factored_policy(ATOMIC_ACTIONS, look_ahead, uniform=True)
        # Connect distribution
        connect_dist = generate_connect_distribution(look_ahead)
        # Evaluation of the algorithms
        test = Evaluation(simu_type, parsed_ins, uniform_policy, look_ahead, log_file, ins_name, param_path)
        for epid in epids:
            for em_num in em_steps:
                rlt = test.simulation(parallel, epid, parsed_ins, alg_choice, connect_dist, em_num)
                alg_mean = rlt[0]
                alg_std = rlt[1]
                update_rlt(alg_choice, look_ahead, s_num, a_num, alg_mean, alg_std, output_log, simu_type)


def main(argv):
    '''
    Multiple algorithms for planning as inference
    '''
    ins_file = argv[1]
    alg_choice = argv[2]
    simu_type = argv[0]
    param_path = argv[3]

    with open(param_path, 'r') as params_file:
        params = json.load(params_file)
        epids = params['epids']
        em_steps = params['epids']
        look_ahead_depths = params['look_ahead_depths']

    ins_path = '../spudd_sperseus/' + ins_file
    log_dir = '../logs/' + alg_choice
    if (not os.path.isdir(log_dir)):
        os.makedirs(log_dir, exist_ok=True)
    # Initialization
    algs_result_comparison = ins_file[:-6] + '_' + alg_choice + '_rlt.json'
    log_path = log_dir + '/' + ins_file[:-6] + '_' + alg_choice + '_'

    output_dir = '../results/' + alg_choice
    if (not os.path.isdir(output_dir)):
        os.makedirs(output_dir, exist_ok=True)
    output_log = output_dir + '/' + algs_result_comparison


    # Simulation
    run(simu_type, alg_choice, ins_path, ins_file, log_path, epids, em_steps, look_ahead_depths, output_log, param_path)


if __name__ == "__main__":
    p = configargparse.ArgParser(default_config_files=['../conf/default.conf'])
    p.add('-i', '--instance', nargs='+', help='full spudd file name of the instance')
    p.add('-a', '--algorithm', nargs='+', help='algorithm name shortcut, including: mfvi_bwd, mfvi_fwd, csvi_bwd, csvi_fwd, bwd_lbp, fwd_lbp, exp_mfvi_bwd, exp_csvi_bwd, mfvi_bwd_noS, demo_mfvi_bwd, demo_mfvi_bwd_noS, demo_mfvi_bwd_med, demo_mfvi_bwd_main_all, demo_csvi_bwd, random.')
    p.add('-s', '--simu_type', help=('simulation type, only support cost'))
    p.add('-app', '--alg_param_path', help=('relative path to algorithm parameters configuration'))
    p.add('-vpp', '--vis_param_path', help=('relative path to visulization parameters configuration'))

    args = p.parse_args()
    simu = args.simu_type
    pp = args.alg_param_path
    for ins in args.instance:
        for alg in args.algorithm:
            main([simu, ins, alg, pp])