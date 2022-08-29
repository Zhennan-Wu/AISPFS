import json
import os
import sys
import fnmatch


def rlt_transfer(alg, input_dir='../SOGBOFA/bin', output_dir='../results', timeout=50000):
    file_format = ''
    if (alg == 'sogbofa'):
        file_format = 'L_C_SOGBOFA_{}*_Score'.format(timeout)
        split_start = 4
    elif (alg == 'sogorigin'):
        file_format = 'SOGBOFA_Original_{}*_Score'.format(timeout)
        split_start = 3
    else:
        raise Exception('Algorithm parameter {} not handled'.format(alg))

    for f_name in os.listdir(input_dir):
        if fnmatch.fnmatch(f_name, file_format):
            inst = f_name.split('_')[split_start:-1]
            if (len(inst) == 5):
                domain = inst[0]
            elif (len(inst) == 6):
                domain = '_'.join(inst[0:2])
            elif (len(inst) == 7):
                domain = '_'.join(inst[0:3])
            else:
                raise Exception('Uncaptured domain name {}'.format(inst))

            inst_idx = inst[-1]
            if (inst_idx != '10'):
                inst[-1] = inst[-1][-1]
            inst_name = '_'.join(inst)
            input_path = input_dir + '/' + f_name
            with open(input_path, 'r') as f:
                line = f.readline().replace('/n', '').split(' ')
                mean = -float(line[1])
                std = float(line[-1])
            create_json(alg, mean, std, inst_name, output_dir)

def create_json(alg, mean, std, inst_name, output_dir):
    alg_dir = output_dir + '/{}'.format(alg)
    if (not os.path.isdir(alg_dir)):
        os.makedirs(alg_dir, exist_ok=True)
    result = {"info": {"horizon": [], "s_num": [], "a_num": []}, alg: {"mean": [mean], "std": [std]}}

    ref_path = output_dir + '/random/{}_random_rlt.json'.format(inst_name)
    if (os.path.exists(ref_path)):
        with open (ref_path, 'r') as ref_info:
            ref = json.load(ref_info)
            result["info"]["horizon"] = ref["info"]["horizon"]
            result["info"]["s_num"] = ref["info"]["s_num"]
            result["info"]["a_num"] = ref["info"]["a_num"]

        output_path = alg_dir + '/{}_{}_rlt.json'.format(inst_name, alg)
        with open(output_path, 'w') as output:
            json.dump(result, output)


if __name__ == '__main__':
    alg = sys.argv[1]
    rlt_transfer(alg)