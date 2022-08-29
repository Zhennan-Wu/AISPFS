import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import fnmatch
import matplotlib
import configargparse
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('seaborn')

######################################################################
# Plot Style
######################################################################
# ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']


def collect(domain, alg, simu_type):
    inst_lst = []
    mean_lst = []
    std_lst = []
    rlt_generated = False

    rlt_dir = '../results/{}'.format(alg)
    if (not os.path.isdir(rlt_dir)):
        raise Exception(
            'The algorithm results directory {} does not exists'.format(rlt_dir))
    for f_name in os.listdir(rlt_dir):
        if (fnmatch.fnmatch(f_name, domain + '*.json')):
            # get instance index
            inst_idx = -1
            splitted_n = f_name.split('_')
            for entry in splitted_n:
                if (entry.isdigit()):
                    inst_idx = int(entry)
            f_path = rlt_dir + '/' + f_name
            with open(f_path) as json_file:
                data = json.load(json_file)
                if (len(data[alg]['mean']) > 0):
                    inst_lst.append(inst_idx)
                    if (simu_type == 'cost'):
                        mean_lst.append(float(data[alg]['mean'][0]))
                    else:
                        mean_lst.append(-float(data[alg]['mean'][0]))
                    std_lst.append(float(data[alg]['std'][0]))

    if (len(inst_lst) > 0):
        rlt_generated = True
        zipped = sorted(zip(inst_lst, mean_lst, std_lst))
        inst_lst, mean_lst, std_lst = zip(*zipped)
    return np.asarray(inst_lst), np.asarray(mean_lst), np.asarray(std_lst), rlt_generated


if __name__ == "__main__":

    p = configargparse.ArgParser(default_config_files=['../conf/vis.conf'])
    p.add('-d', '--domain', nargs='+', help='domain name')
    p.add('-a', '--algorithm', nargs='+', help='algorithm name shortcut, including: mfvi_bwd, mfvi_fwd, csvi_bwd, csvi_fwd, bwd_lbp, fwd_lbp, exp_mfvi_bwd, exp_csvi_bwd, mfvi_bwd_noS, demo_mfvi_bwd, demo_mfvi_bwd_noS, demo_mfvi_bwd_med, demo_mfvi_bwd_main_all, demo_csvi_bwd, random.')
    p.add('-s', '--simu_type', help=('simulation type, reward or cost'))
    p.add('-vpp', '--vis_param_path', help=('relative path to visulization parameters configuration'))    

    args = p.parse_args()
    simu_type = args.simu_type
    domains = args.domain
    algs = args.algorithm
    param_path = args.vis_param_path
    
    with open(param_path, 'r') as params_file:
        params = json.load(params_file)
        formats = params['formats']
        fillstyles = params['fillstyles']
        labels = params['labels']
        colors = params['colors']
        plots_name = params['plots_name']
    for domain in domains:
        for alg in algs:
            ins_ary, mean_ary, std_ary, rlt_generated = collect(
                domain, alg, simu_type)
            if (rlt_generated):
                plt.errorbar(ins_ary, mean_ary, std_ary, label=labels[alg], linestyle='dashed',
                            fmt=formats[alg], color=colors[alg], fillstyle=fillstyles[alg], elinewidth=2, linewidth=0.5)

        plt.title('{} Domain Performance'.format(domain.capitalize()), fontweight="bold", fontsize=18)
        plt.xlabel("Instance index", fontweight="bold", fontsize=15)
        plt.ylabel("Cumulative {}".format(simu_type), fontweight="bold", fontsize=15)
        plt.legend(prop={'weight':'bold', 'size': 14})
        # plt.legend()
        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)
        plot_dir = '../results/plots/{}'.format(simu_type)
        if (not os.path.isdir(plot_dir)):
            os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_dir + '/{}_full.pdf'.format(domain), bbox_inches='tight')
        plt.savefig(plot_dir + '/{}_full.png'.format(domain), bbox_inches='tight')
        plt.close()