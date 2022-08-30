import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import fnmatch
import copy
import matplotlib.ticker as ticker
import matplotlib
import configargparse
from utils import record
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
par = {'axes.titleweight':'bold'}
plt.rcParams.update(par)
plt.style.use('seaborn')

######################################################################
# Plot Style
######################################################################
# ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']


def aggre(algs, param_path, plotting = True):
    '''
    Read result json file for plotting purpose.
    '''
    with open(param_path, 'r') as params_file:
        params = json.load(params_file)
        domains = params['domains']
        formats = params['formats']
        fillstyles = params['fillstyles']
        labels = params['labels']
        colors = params['colors']
        plots_name = params['plots_name']

    Domains = []
    mean_lst = {}
    std_lst = {}
    domain_idx = 0
    inst_num = 0
    for alg in algs:
        mean_lst[alg] = {'idx': [], 'val': []}
        std_lst[alg] = {'idx': [], 'val': []}
    for domain in domains:
        domain_idx += inst_num
        d_short = domain.capitalize().split('_')[0][0:5]
        # Domains.append(domain.capitalize())
        Domains.append(d_short)
        # generate random results
        alg = 'random'
        rlt_dir = '../results/{}'.format(alg)
        if (not os.path.isdir(rlt_dir)):
            raise Exception('The algorithm results directory {} does not exists'.format(rlt_dir))
        random_mean = {}
        for f_name in os.listdir(rlt_dir):
            if (fnmatch.fnmatch(f_name, domain + '*.json')):
                # get instance index
                inst_idx = -1
                splitted_n = f_name.split('_')
                for entry in splitted_n:
                    if (entry.isdigit()):
                        inst_idx = entry
                f_path = rlt_dir + '/' + f_name
                with open(f_path) as json_file:
                    data = json.load(json_file)
                    if (len(data[alg]['mean']) > 0):
                        random_mean[inst_idx] = float(data[alg]['mean'][0])

        inst_num = 0
        idx_dict = {}
        for alg in algs:
            rlt_dir = '../results/{}'.format(alg)
            inst_count = 0
            if (not os.path.isdir(rlt_dir)):
                raise Exception('The algorithm results directory {} does not exists'.format(rlt_dir))
            for f_name in os.listdir(rlt_dir):
                if (fnmatch.fnmatch(f_name, domain + '*.json')):
                    # get instance index
                    inst_count += 1
                    inst_idx = -1
                    splitted_n = f_name.split('_')
                    # print(splitted_n)
                    for entry in splitted_n:
                        if (entry.isdigit()):
                            inst_idx = entry
                            if (not inst_idx in idx_dict.keys()):
                                idx_dict[inst_idx] = domain_idx + int(inst_idx)
                    if (inst_idx != -1):
                        f_path = rlt_dir + '/' + f_name
                        with open(f_path) as json_file:
                            data = json.load(json_file)
                            if (len(data[alg]['mean']) > 0):
                                mean_lst[alg]['idx'].append(idx_dict[inst_idx])
                                std_lst[alg]['idx'].append(idx_dict[inst_idx])
                                mean_lst[alg]['val'].append((random_mean[inst_idx] - float(data[alg]['mean'][0]))/abs(random_mean[inst_idx]))
                                # mean_lst[alg]['val'].append(max((random_mean[inst_idx] - float(data[alg]['mean'][0]))/abs(random_mean[inst_idx]), -1))
                                std_lst[alg]['val'].append(float(data[alg]['std'][0])/abs(random_mean[inst_idx]))
            inst_num = max(inst_num, inst_count)

    if (plotting):
        fig, ax = plt.subplots()
        for alg in mean_lst.keys():
            insts = copy.deepcopy(mean_lst[alg]['idx'])
            means = copy.deepcopy(mean_lst[alg]['val'])
            stds = copy.deepcopy(std_lst[alg]['val'])
            zipped = sorted(zip(insts, means, stds))
            inst_sorted, mean_sorted, std_sorted= zip(*zipped)

            ins_ary = np.asarray(inst_sorted)
            mean_ary = np.asarray(mean_sorted)
            std_ary = np.asarray(std_sorted)

            ax.errorbar(ins_ary, mean_ary, std_ary, label=labels[alg], linestyle='dashed', fmt=formats[alg], color = colors[alg], fillstyle = fillstyles[alg], elinewidth=2, linewidth=0.5)

        xposition = [0.5, 10.5, 20.5, 30.5, 40.5, 50.5, 60.5]
        for xc in xposition:
            ax.axvline(x=xc, color='black', linestyle='--', linewidth=0.3)
        
        x = [5.5, 15.5, 25.5, 35.5, 45.5, 55.5]
        plt.xticks(x, Domains, fontweight='bold', fontsize=14)
        # fig.autofmt_xdate()

        if (algs[-1]!= 'random'):
            print('plot title will be wrong')
        if (len(alg) == 1):
            ax.set_title('{}'.format(plots_name[algs[0]]), fontdict={'fontsize': 18})
        elif (len(algs) == 4):
            ax.set_title('{} VS {} VS {}'.format(labels[algs[0]], labels[algs[1]], labels[algs[2]]), fontdict={'fontsize': 18})
        elif (len(algs) == 3):
            ax.set_title('{} VS {}'.format(labels[algs[0]], labels[algs[1]]), fontdict={'fontsize': 18})
        else:
            ax.set_title('{}'.format(algs), fontdict={'fontsize': 18})

        ax.set_ylabel("Score", fontweight='bold', fontsize=15)
        # ax.set_ylim([-2, 4])
        ax.set_ylim([-1.5, 3.5])
        legend_properties = {'weight':'bold', 'size': 14}
        ax.legend(loc='upper left', prop = legend_properties)
        plot_dir = '../results/plots'
        if (not os.path.isdir(plot_dir)):
            os.makedirs(plot_dir, exist_ok=True)
        ax.tick_params(axis='y', labelsize=14)
        fig.savefig(plot_dir + '/{} summary.pdf'.format(algs), bbox_inches='tight')
        fig.savefig(plot_dir + '/{} summary.png'.format(algs), bbox_inches='tight')
        plt.close()

    return mean_lst, std_lst


def calc_win(comp_algs, mean_lst):
    '''
    Summary of algorithm performance.
    '''
    win_count = {}
    log_path = '../results/alg_comp.txt'
    for rdm_idx, inst_idx in enumerate(mean_lst['random']['idx']):
        # best_perform = mean_lst['random']['val'][rdm_idx]
        # best_alg = 'random'
        best_perform = -10
        best_alg = 'none'

        # only count the comparison result if all algorithm in comparison have
        # results on the instances
        comp_status = True
        for alg in comp_algs:
            if (inst_idx in mean_lst[alg]['idx']):
                alg_idx = mean_lst[alg]['idx'].index(inst_idx)
                if (mean_lst[alg]['val'][alg_idx] > best_perform):
                    best_perform = mean_lst[alg]['val'][alg_idx]
                    best_alg = alg
                elif (mean_lst[alg]['val'][alg_idx] == best_perform):
                    if (alg != 'random'):
                        best_perform = mean_lst[alg]['val'][alg_idx]
                        best_alg = alg
            else:
                comp_status = False
        if (comp_status):
            # if (best_alg == 'random'):
            #     print('random', best_perform)
            #     for alg in comp_algs:
            #         print(alg, mean_lst[alg]['val'][mean_lst[alg]['idx'].index(inst_idx)])
            if (best_alg in win_count.keys()):
                win_count[best_alg] = win_count[best_alg] + 1
            else:
                win_count[best_alg] = 1

    record('Comparison among {}\n'.format(comp_algs), log_path)
    record(win_count, log_path)


if __name__ == "__main__":
    p = configargparse.ArgParser(default_config_files=['../conf/vis.conf'])
    p.add('-d', '--domain', nargs='+', help='domain name')
    p.add('-a', '--algorithm', nargs='+', help='algorithm name shortcut, including: mfvi_bwd, mfvi_fwd, csvi_bwd, csvi_fwd, bwd_lbp, fwd_lbp, exp_mfvi_bwd, exp_csvi_bwd, mfvi_bwd_noS, demo_mfvi_bwd, demo_mfvi_bwd_noS, demo_mfvi_bwd_med, demo_mfvi_bwd_main_all, demo_csvi_bwd, random.')
    p.add('-s', '--simu_type', help=('simulation type, only support cost'))
    p.add('-vpp', '--vis_param_path', help=('relative path to visulization parameters configuration'))

    args = p.parse_args()
    algs = args.algorithm
    param_path = args.vis_param_path

    mean_lst, std_lst = aggre(algs, param_path)

    calc_win(algs, mean_lst)
