import copy
import math
import itertools
import numpy as np

class SPUDD_Parser:
    def __init__(self, file):
        #-----------------------------------------------------------------------
        # Inner variables
        #-----------------------------------------------------------------------

        self.domain = file
        self.name = file.replace('/', '.').split('.')[-2]

        # Transition distribution, hierarchical dictionary, top key to be action, second layer is the state var under consideration, bottom value is next time-step probability
        self.trans_dict = {}
        # { <action>:
        #   { <child_svar>:
        #        { <parent_svar1>:
        #                   { <ps1_val1>:
        #                       { <parent_svar2>:
        #                           ...
        #                           { <last_p_val1>:
        #                               [(list of ps in this branch), val of ps in this branch, child_prob1]}}}}
        #       ....
        #                   { <ps1_val2>:
        #                       {...}}}}

        # Flat version, for each action key, for each state variable key, store a list of [parent state variable, parent state value, distribution]
        # as the value
        self.enum_table = {}
        # { <action>:
        #       {child_svar:
        #           [(parent_svar-case1), ps_val-case1, (prob0, prob1)], [(parent_svar-case2), ps_val-case2, (prob0, prob1)], ...}}

        # Compact vectorized version of transition dynamics
        self.vec_trans = {}
        # { <child_svar1:
        #   {'parents': [state var1, state var2, ...], 'space': [state grounding1, state grounding2, ...], actions':[action1, action2, ...], 'trans': np.array(prob of child_svar1 = 1)_{#state_vars x #actions}}
        #   ...}

        # Full transition table of dependent state and action variables
        self.pseudo_trans = {}
        # { <child_svar1>:
        #       { <parent_s_val1>:
        #           { <action_val1>: (prob0, prob1)}
        #           { <action_val2>: (prob0, prob1)}
        #           ...
        #       { <parent_s_val2>:
        #           { <action_val1>: (prob0, prob1)}
        #           ...}
        #       ...}}

        # Store noop probability for later use
        self.noop_table = {}
        # { <child_svar>:
        #       { <parent_sval1>: (prob0, prob1)
        #         <parent_sval2>: (prob0, prob1)}
        # ...}

        # a list of maximum reward for each action choice
        self.r_max_lst = []
        self.r_max_lst_unfactored = []
        # a list of minimum reward for each action choice
        self.r_min_lst = []
        self.r_min_lst_unfactored = []

        #-----------------------------------------------------------------------
        # Output variables
        #-----------------------------------------------------------------------

        # State initialization table, key to be state variable, value to be a two-element list of [prob0, prob1]
        self.init_state = {}
        # { <state_var1>: [prob0, prob1]
        #   <state_var2>: [prob0, prob1]
        #   ...}

        # A list of all state variables
        self.state_vars = []
        # A list of all action variables
        self.action_vars = []
        # A list of all valid atomic actions
        self.atomic_action_lst = []
        # A list of all valid action groundings
        self.valid_action_lst = []

        # A list of valid [atomic action, action grounding] pairs
        self.valid_action_pairs = []

        # Cost (reward) dictionary w.r.t state, optional action default cost stored in "extra" key
        self.reward_table = {}
        # { <atomic_action>:
        #   { "extra": intrinsic_action_cost
        #     "cases": [{case1: reward}, {case2: reward}, ...]
        #     "parents":[[case1 parent state vars], [case2 parent state vars], ...]
        #     "enum": [[{'svar1': sval1, ..., 'val': rwd}, ...], [case2], ...]
        #     "value": action grounding of the atomic action}
        #   ...}

        # Vectorized cost (reward) table
        # [{'state_vars': [s_var1, s_var2, ...], 'actions': [act_1, act_2, ...], 'rwd': np.array(rwd)_{#states x # actions}},
        # {case2}, ...]
        self.vec_rwds = []
        self.vec_raw_rwds = []
        self.exp_vec_rwds = []


        # Dependency of state, key to be state variables, value to be a list of parent state variables
        self.state_dependency = {}
        # { <child_svar>': [parent_svar1, parent_svar2, ...]
        #   ...}

        # Enumerate all different choice of transition
        # for each child state variable as key of all dictionaries,
        # store the corresponding
        self.parent_s_table = {}
        #   parent state value  -> list: parent_s_table[<child_svar>]
        self.parent_a_table = {}
        #   action grounding -> list: parent_a_table[<child_svar>]
        self.parent_a_ref = {}
        #   atomic action choice -> list: parent_a_ref[<child_svar>]
        self.child_prob_table = {}
        #   transition probability -> list: child_prob_table[<child_svar]
        self.compact_trans_dict = {}
        #   [ps_val, a_grounding, atomic_act, prob] -> list: compact_trans_dict[<child_svar>]

        # A list of joint distribution of transition, with each entry to be a dictionary consisting of  "row_var", "row_val", "column_val", "table"
        self.joint_trans_table = []
        # [ {"row_var": [parent_state_vars, child_state_var], "row_val": [grounding_of_row_vars], "column_val": [action_vals], "table": np.array(prob)}, ...]

        # A list of joint distribution of transition, with each entry to be a dictionary consisting of  "row_var", "row_val", "column_val", "table"
        self.joint_rwd_table = []
        self.joint_raw_rwd_table = []
        self.joint_exp_rwd_table = []
        # [ {"row_var": [state_vars, rwd_var], "row_val": [grounding_of_row_vars], "column_val": [action_vals], "table": np.array(prob)}, ...]

        self.trans_group = {}
        self.rwd_group = {}

        self.discount = -1
        self.horizon = -1

        # reward normalization hack
        self.bound = 0
        self.bound_record = {}

        self.r_max = 0
        self.r_max_unfactored = 0
        self.r_min = 0
        self.r_min_unfactored = 0
        self.epsilon = 1e-3
        # initialize reward cases with action cost, thus to be 1
        self.rwd_cases = 1

        self.val_candid = []
        self.max_candid = []
        self.min_candid = []

        self._parse(file)
        self._generate_valid_action_lst()
        self._action_val_collect()
        self._normalize_reward()
        self._init_vec_trans()
        self._vectorize_reward()
        self._vectorize_exp_reward()
        self._extract_compact_transition()
        self._generate_pseudo_trans()
        self.generate_joint_table()

    def _list_to_string(self, lst):
        string = ''.join(lst)
        return string

    def _find_atomic_action(self, a_var_lst):
        '''
        From action variable recover the atomic action.
        '''
        atomic_grounding = ''
        for a_var in self.action_vars:
            if (a_var in a_var_lst):
                atomic_grounding = atomic_grounding + '1'
            else:
                atomic_grounding = atomic_grounding + '0'
        idx = self.valid_action_lst.index(atomic_grounding)
        return self.atomic_action_lst[idx]

    def _generate_valid_action_lst(self):
        '''
        Encode all actions into binary form.
        '''
        for aref in self.reward_table.keys():
            act = ['0']*len(self.action_vars)
            self.atomic_action_lst.append(aref)
            factored_a = aref.split("___")
            for fa in factored_a:
                if (fa == 'noop'):
                    continue
                else:
                    for idx, var in enumerate(self.action_vars):
                        if (var == fa):
                            act[idx] = '1'
            self.valid_action_lst.append(''.join(act))
            self.valid_action_pairs.append([aref, ''.join(act)])

    def _generate_full_space(self, num):
        '''
        Generate state (action) space, represented as python list.
        :param num: number of state (action) variables, represented as int.
        '''
        if (num > 0):
            size = pow(2, num)
            space = []
            for i in range(size):
                b = bin(i)[2:]
                l = len(b)
                b = str(0)*(num - l) + b
                space.append(b)
        else:
            space = ['']
        return space

    def _soften(self, val):
        '''
        Increase the numerical stability of the probabilities.
        '''
        eps = 1e-4
        if (val >= 1.- eps):
            soften = 1. - eps
        elif (val <= eps):
            soften = eps
        else:
            soften = val
        return soften

    def _parse(self, file):
        '''
        Fill in
        - self.init_state
        - self.action_vars
        - self.trans_dict
        - self.reward_table
        - self.state_dependency
        - self.discount
        - self.horizon
        - self.enum_table
        '''
        with open(file, 'r') as f:
            var_block = False
            init_block = False
            action_block = False
            cost_block = False
            lines = f.readlines()
            # temporal storage for transition dictionary creation
            state_tree = []
            curr_act = ''
            curr_state = ''
            debug = False
            reward_var_tree = []
            reward_enum_tree = []
            reward_factor_tree = []

            for line in lines:
                if (not var_block and not init_block and not action_block):
                    if (line.replace('\n', '').replace('(', '').replace(' ', '') == 'variables'):
                        var_block = True
                        continue

                    if (line.replace('\n', '').replace('[', '').replace(' ', '').replace('*', '') == 'init'):
                        init_block = True
                        continue

                    a_title = line.replace('\n', '').split(' ')
                    if (a_title[0] == 'action'):
                        action_block = True
                        curr_act = a_title[1]
                        self.trans_dict[curr_act] = {}
                        self.enum_table[curr_act] = {}
                        self.reward_table[curr_act] = {}
                        self.reward_table[curr_act]['extra'] = 1e-8
                        self.bound_record[curr_act] = 1e-8
                        self.reward_table[curr_act]['cases'] = []
                        self.reward_table[curr_act]['parents'] = []
                        self.reward_table[curr_act]['enum'] = []

                        factored_act = curr_act.split('___')
                        for f_act in factored_act:
                            if (not f_act in self.action_vars):
                                if (not f_act == 'noop'):
                                    self.action_vars.append(f_act)
                        continue

                    raw = line.replace('\n', '').split(' ')
                    if (raw[0] == 'discount'):
                        self.discount = float(raw[1])
                    elif (raw[0] == 'horizon'):
                        self.horizon = int(raw[1])
                    else:
                        continue

                c_title = line.replace('\n', '').replace('\t', '').replace('[', '').split(' ')
                if (c_title[0] == 'cost'):
                    cost_block = True
                    continue

                if (var_block):
                    if (line.replace('\n', '') == ')'):
                        var_block = False
                        continue

                    var = line.replace("(", '').replace(")", '').replace("\n", '').replace('\t', '').split(' ')
                    if (var[1] != 'true' or var[2] != 'false'):
                        raise Exception('Not binary variables')
                    else:
                        self.init_state[var[0]] = []
                        child_state = var[0]+"'"
                        self.state_dependency[child_state] = []

                elif (init_block):
                    if (line.replace('\n', '') == ']'):
                        init_block = False
                        self.state_vars = list(self.init_state.keys())
                        continue

                    init_val = line.replace("(", '').replace(")", '').replace("\n", '').replace('\t', '').split(' ')
                    true_val = -1
                    false_val = -1
                    for idx, v in enumerate(init_val):
                        if (v == 'true'):
                            true_val = float(init_val[idx+1])
                        elif (v == 'false'):
                            false_val = float(init_val[idx+1])
                        else:
                            continue
                    self.init_state[init_val[0]] = [false_val, true_val]

                elif (action_block and not cost_block):
                    if (line.replace('\n', '') == 'endaction'):
                        action_block = False
                        continue

                    raw = line.replace('\n', '').replace('\t', '').replace('(', '').replace(')', '').split(' ')
                    # The first line below the state variable name in SPUDD has an additional ' ',
                    # so use it to distinguish between state variable declaration line and grounding line.
                    if (len(raw) == 1 and raw[0] in self.init_state.keys()):
                        val_record = []
                        var_record = []
                        curr_state = raw[0]
                        self.enum_table[curr_act][curr_state] = []
                        future_state = curr_state + "'"
                        self.trans_dict[curr_act][curr_state] = {}
                    elif (len(raw) == 2):
                        # state variables declaration line
                        if (raw[0] in self.init_state.keys()):
                            if (not raw[0] in self.state_dependency[future_state]):
                                if (not raw[0][-1] == "'"):
                                    self.state_dependency[future_state].append(raw[0])
                            state_node = {raw[0]: {}}
                            state_tree.append(state_node)
                            var_record.append(raw[0])
                            if (debug):
                                print(state_tree)
                        # cases where state variables have no parent
                        elif (raw[0].replace("'", '') in self.init_state.keys()):
                            state_node = {raw[0]: {}}
                            state_tree.append(state_node)
                        # leaf probability line
                        else:
                            record = self._list_to_string(val_record)
                            ref_record = tuple(var_record)
                            merging = True
                            if (raw[0] == 'true'):
                                for key in state_tree[-1].keys():
                                    state_tree[-1][key]['true'] = [ref_record, record, self._soften(float(raw[1]))]
                                    self.enum_table[curr_act][curr_state].append([ref_record, record, (1-self._soften(float(raw[1])), self._soften(float(raw[1])))])
                                if (debug):
                                    print(state_tree)
                            elif (raw[0] == 'false'):
                                for key in state_tree[-1].keys():
                                    state_tree[-1][key]['false'] = [ref_record, record, self._soften(float(raw[1]))]
                                if (debug):
                                    print(state_tree)
                            else:
                                raise Exception('Uncaptured edge case {}, keys {}'.format(raw, self.init_state.keys()))

                            while (merging):
                                for key in state_tree[-1].keys():
                                    if (len(state_tree[-1][key]) == 2):
                                        if (var_record):
                                            # pop completed parent variable value assignment
                                            if (var_record[-1] == key):
                                                var_record.pop()
                                        if (len(state_tree) == 1):
                                            self.trans_dict[curr_act][curr_state] = state_tree.pop()
                                            merging = False
                                        else:
                                            # recursively pop out completed branches
                                            for key1 in state_tree[-2].keys():
                                                for key2 in state_tree[-2][key1].keys():
                                                    if (isinstance(state_tree[-2][key1][key2], str)):
                                                        state_tree[-1][key1][key2] = state_tree.pop()
                                                        val_record.pop()

                                        if (debug):
                                            print(state_tree)
                                    else:
                                        merging = False
                    elif (len(raw) == 3):
                        if (raw[0] == 'true' or raw[0] == 'false'):
                            if (raw[0] == 'true'):
                                val_record.append('1')
                            else:
                                val_record.append('0')

                            if (not raw[1][-1] == "'"):
                                var_record.append(raw[1])

                            for key in state_tree[-1].keys():
                                state_tree[-1][key][raw[0]] = raw[1]
                            if (not raw[1] in self.state_dependency[future_state]):
                                if (not raw[1][-1] == "'"):
                                    self.state_dependency[future_state].append(raw[1])
                            state_tree.append({raw[1]: {}})
                            if (debug):
                                print(state_tree)
                elif (cost_block):
                    raw = line.replace('\n', '').replace('\t', '')
                    if ( raw == ']'):
                        self.reward_table[curr_act]['parents'].append(reward_var_tree)
                        reward_var_tree = []
                        self.r_max_unfactored += max(self.max_candid)
                        self.r_min_unfactored += min(self.min_candid)

                        if (len(reward_factor_tree) > 0):
                            self.reward_table[curr_act]['enum'].append(reward_factor_tree)
                            reward_factor_tree = []
                        
                        cost_block = False
                        self.r_max_lst.append(self.r_max)
                        self.r_max_lst_unfactored.append(self.r_max_unfactored)
                        self.r_min_lst.append(self.r_min)
                        self.r_min_lst_unfactored.append(self.r_min_unfactored)
                        self.r_max = 0
                        self.r_min = 0
                        self.r_max_unfactored = 0
                        self.r_min_unfactored = 0
                        self.max_candid = []
                        self.min_candid = []
                        continue

                    raw = line.replace('\n', '').replace('\t', '').replace('(', '').replace(')', '').split(' ')
                    if (len(raw) == 1):
                        self.reward_table[curr_act]['extra'] += -float(raw[0])
                        self.bound_record[curr_act] += abs(float(raw[0]))
                        self.bound = max(self.bound, self.bound_record[curr_act])
                        self.r_max_unfactored += -float(raw[0])
                        self.r_min_unfactored += -float(raw[0])
                        self.r_max = max(self.r_max, -float(raw[0]))
                        self.r_min = min(self.r_min, -float(raw[0]))
                    elif (len(raw) == 2):
                        if (raw[0] in self.init_state.keys()):
                            if (len(self.max_candid) > 0):
                                self.r_max_unfactored += max(self.max_candid)
                                self.r_min_unfactored += min(self.min_candid)
                                self.max_candid = []
                                self.min_candid = []
                            if (len(reward_var_tree) > 0):
                                self.reward_table[curr_act]['parents'].append(reward_var_tree)
                                reward_var_tree = []

                            if (len(reward_factor_tree) > 0):
                                self.reward_table[curr_act]['enum'].append(reward_factor_tree)
                                reward_factor_tree = []
                            if (len(reward_enum_tree) > 0):
                                raise ValueError('Recursive enumerate reward wrong')                            
                            reward_var_tree.append(raw[0])

                            reward_enum_tree.append({'val': 0, raw[0]: '0'})

                            state_node = {raw[0]: {}}
                            state_tree.append(state_node)
                        else:
                            merging = True
                            undecided_var = list(reward_enum_tree[-1])[-1]
                            reward_enum_tree[-1]['val'] = -float(raw[1])
                            if (raw[0] == 'true'):
                                reward_enum_tree.append(copy.deepcopy(reward_enum_tree[-1]))
                                reward_enum_tree[-1][undecided_var] = '1'
                                reward_enum_tree[-1]['val'] = -float(raw[1])
                                reward_factor_tree.append(reward_enum_tree.pop())
                                for key in state_tree[-1].keys():
                                    state_tree[-1][key]['true'] = -float(raw[1])

                                    self.bound = max(self.bound, abs(float(raw[1])))

                                    self.val_candid.append(-float(raw[1]))
                                    self.min_candid.append(-float(raw[1]))
                                    self.max_candid.append(-float(raw[1]))

                                    if (len(self.val_candid) == 2):
                                        max_val = max(self.max_candid)
                                        self.max_candid = [max_val]
                                        
                                        min_val = min(self.min_candid)
                                        self.min_candid = [min_val]

                                        self.r_max = max(self.r_max, max(self.val_candid))
                                        self.r_min = min(self.r_min, min(self.val_candid))
                                        self.val_candid = []
                                if (debug):
                                    print(state_tree)
                            elif (raw[0] == 'false'):
                                reward_enum_tree[-1][undecided_var] = '0'
                                reward_enum_tree[-1]['val'] = -float(raw[1])
                                reward_factor_tree.append(reward_enum_tree.pop())
                                for key in state_tree[-1].keys():
                                    state_tree[-1][key]['false'] = -float(raw[1])
                                    self.bound = max(self.bound, abs(float(raw[1])))

                                    self.val_candid.append(-float(raw[1]))
                                    self.min_candid.append(-float(raw[1]))
                                    self.max_candid.append(-float(raw[1]))

                                    if (len(self.val_candid) == 2):
                                        max_val = max(self.max_candid)
                                        self.max_candid = [max_val]
                                        
                                        min_val = min(self.min_candid)
                                        self.min_candid = [min_val]

                                        self.r_max = max(self.r_max, max(self.val_candid))
                                        self.r_min = min(self.r_min, min(self.val_candid))
                                        self.val_candid = []
                                if (debug):
                                    print(state_tree)
                            else:
                                raise Exception('Uncaptured edge case {}, keys {}'.format(raw, self.init_state.keys()))
                            while (merging):
                                for key in state_tree[-1].keys():
                                    if (len(state_tree[-1][key]) == 2):
                                        if (len(state_tree) == 1):
                                            self.reward_table[curr_act]['cases'].append(state_tree.pop())
                                            merging = False
                                        else:
                                            for key1 in state_tree[-2].keys():
                                                for key2 in state_tree[-2][key1].keys():
                                                    if (isinstance(state_tree[-2][key1][key2], str)):
                                                        state_tree[-1][key1][key2] = state_tree.pop()
                                        if (debug):
                                            print(state_tree)
                                    else:
                                        merging = False
                    elif (len(raw) == 3):
                        if (raw[0] == 'true' or raw[0] == 'false'):
                            undecided_var = list(reward_enum_tree[-1])[-1]
                            if (raw[0] == 'true'):
                                reward_enum_tree.append(copy.deepcopy(reward_enum_tree[-1]))
                                reward_enum_tree[-1][undecided_var
                                ] = '1'
                            else:
                                reward_enum_tree[-1][undecided_var
                                ] = '0'
                            reward_enum_tree[-1][raw[1]] = '0'

                            for key in state_tree[-1].keys():
                                state_tree[-1][key][raw[0]] = raw[1]
                            state_tree.append({raw[1]: {}})
                            if (raw[1] not in reward_var_tree):
                                reward_var_tree.append(raw[1])
                            if (debug):
                                print(state_tree)
                    else:
                        raise Exception('Uncaptured cases {}'.format(raw))

    def _action_val_collect(self):
        for act in self.reward_table.keys():
            val = ['0']*len(self.action_vars)
            factored_act = act.split('___')
            for f_act in factored_act:
                if (f_act == 'noop'):
                    self.reward_table[act]['value'] = ''.join(val)
                else:
                    for idx, act_choice in enumerate(self.action_vars):
                        if (f_act == act_choice):
                            val[idx] = '1'
                            continue
            self.reward_table[act]['value'] = ''.join(val)

    def _vectorize_reward(self):
        '''
        Create vectorized reward representation
        - a_rwd: one case of reward table
        - self.vec_rwds: a list of reward assignment cases
        - parameters
            state_vars: a list of reward dependent state variables
            actions: a list of all valid actions
            val: an numpy array of cost in each case for each action
        '''
        # adding intial action cost (reward) term to the cost (reward) list
        a_rwd = {'state_vars': [], 'reward': 'pr1', 'space': ['0', '1']}
        a_rwd['actions'] = self.atomic_action_lst
        a_rwd['val'] = np.zeros((2, len(a_rwd['actions'])))

        a_raw = {'state_vars': [], 'reward': 'pr1', 'space': ['0', '1']}
        a_raw['actions'] = self.atomic_action_lst
        a_raw['val'] = np.zeros((2, len(a_rwd['actions'])))        
        # assume different actions have the same number of cases
        for _ in self.reward_table['noop']['cases']:
            self.vec_rwds.append(dict())
            self.vec_raw_rwds.append(dict())
            self.rwd_cases += 1

        for act in self.reward_table.keys():
            idx = a_rwd['actions'].index(act)

            raw = (self.reward_table[act]['extra'] - (self.r_min - self.epsilon))/(self.r_max - (self.r_min - self.epsilon))
            prob = self._soften(raw)
            a_rwd['val'][0, idx] = 1 - prob
            a_rwd['val'][1, idx] = prob

            a_raw['val'][1, idx] = self.reward_table[act]['extra']        

        vec_init = False

        for a_idx, act in enumerate(self.reward_table.keys()):
            if (a_idx != 0):
                vec_init = True
            for c_idx, _ in enumerate(self.reward_table[act]['cases']):
                self.vec_rwds[c_idx]['state_vars'] = copy.deepcopy(self.reward_table[act]['parents'][c_idx])
                self.vec_rwds[c_idx]['actions'] = self.atomic_action_lst
                self.vec_rwds[c_idx]['reward'] = 'pr' + str(c_idx + 2)

                self.vec_raw_rwds[c_idx]['state_vars'] = copy.deepcopy(self.reward_table[act]['parents'][c_idx])
                self.vec_raw_rwds[c_idx]['actions'] = self.atomic_action_lst
                self.vec_raw_rwds[c_idx]['reward'] = 'pr' + str(c_idx + 2)

                space = self._enumerate_space(len(self.vec_rwds[c_idx]['state_vars']) + 1)
                self.vec_rwds[c_idx]['space'] = space
                self.vec_raw_rwds[c_idx]['space'] = space
                s_dim = len(space)
                a_dim = len(self.vec_rwds[c_idx]['actions'])
                if (not vec_init):
                    self.vec_rwds[c_idx]['val'] = np.zeros((s_dim, a_dim))
                    for i in range(s_dim):
                        if (2*i < s_dim):
                            self.vec_rwds[c_idx]['val'][2*i] = np.ones(a_dim)
                    self.vec_raw_rwds[c_idx]['val'] = np.zeros((s_dim, a_dim))

                for grounding in self.reward_table[act]['enum'][c_idx]:

                    full_s = self._recover_reward_dependence(self.vec_rwds[c_idx]['state_vars'], grounding)

                    for s_grounding in full_s:
                        s0_idx = space.index(s_grounding + '0')

                        raw = (grounding['val'] - (self.r_min - self.epsilon))/(self.r_max - (self.r_min - self.epsilon))
                        prob = self._soften(raw)

                        self.vec_rwds[c_idx]['val'][s0_idx, a_idx] = 1. - prob

                        s1_idx = space.index(s_grounding + '1')
                        self.vec_rwds[c_idx]['val'][s1_idx, a_idx] = prob
                        self.vec_raw_rwds[c_idx]['val'][s1_idx, a_idx] = grounding['val']

        self.vec_rwds.insert(0, a_rwd)
        self.vec_raw_rwds.insert(0, a_raw)

    def _vectorize_exp_reward(self):
        '''
        Create vectorized reward representation
        - a_rwd: one case of reward table
        - self.vec_rwds: a list of reward assignment cases
        - parameters
            state_vars: a list of reward dependent state variables
            actions: a list of all valid actions
            val: an numpy array of cost in each case for each action
        '''
        # adding intial action cost (reward) term to the cost (reward) list
        a_rwd = {'state_vars': [], 'reward': 'pr1', 'space': ['0', '1']}
        a_rwd['actions'] = self.atomic_action_lst
        a_rwd['val'] = np.zeros((2, len(a_rwd['actions'])))
        for act in self.reward_table.keys():
            idx = a_rwd['actions'].index(act)
            raw = math.exp(self.reward_table[act]['extra'])/math.exp(self.r_max)
            prob = self._soften(raw)
            a_rwd['val'][0, idx] += 1. - prob
            a_rwd['val'][1, idx] += prob

        # assume different actions have the same number of cases
        for _ in self.reward_table['noop']['cases']:
            self.exp_vec_rwds.append(dict())

        vec_init = False
        for a_idx, act in enumerate(self.reward_table.keys()):
            if (a_idx != 0):
                vec_init = True
            for c_idx, _ in enumerate(self.reward_table[act]['cases']):
                self.exp_vec_rwds[c_idx]['state_vars'] = copy.deepcopy(self.reward_table[act]['parents'][c_idx])
                self.exp_vec_rwds[c_idx]['actions'] = self.atomic_action_lst
                self.exp_vec_rwds[c_idx]['reward'] = 'pr' + str(c_idx + 2)

                space = self._enumerate_space(len(self.vec_rwds[c_idx+1]['state_vars']) + 1)
                self.exp_vec_rwds[c_idx]['space'] = space
                s_dim = len(space)
                a_dim = len(self.vec_rwds[c_idx]['actions'])
                if (not vec_init):
                    self.exp_vec_rwds[c_idx]['val'] = np.zeros((s_dim, a_dim))
                    for i in range(s_dim):
                        if (2*i < s_dim):
                            self.exp_vec_rwds[c_idx]['val'][2*i] = np.ones(a_dim)
                for grounding in self.reward_table[act]['enum'][c_idx]:
                    full_s = self._recover_reward_dependence(self.vec_rwds[c_idx+1]['state_vars'], grounding)
                    for s_grounding in full_s:
                        s_idx = space.index(s_grounding + '0')
                        raw = math.exp(grounding['val'])/math.exp(self.r_max)
                        prob = self._soften(raw)

                        self.exp_vec_rwds[c_idx]['val'][s_idx, a_idx] = 1. - prob

                        s_idx = space.index(s_grounding + '1')
                        self.exp_vec_rwds[c_idx]['val'][s_idx, a_idx] = prob

        self.exp_vec_rwds.insert(0, a_rwd)

    def _recover_reward_dependence(self, vars, val_dict):
        '''
        generate a list of state space that satisfies the "val_dict" information
        '''
        full_s_lst = ['']
        for var in vars:
            if (var in val_dict.keys()):
                if (val_dict[var] != '2'):
                    for idx, _ in enumerate(full_s_lst):
                        full_s_lst[idx] = full_s_lst[idx] + val_dict[var]
                else:
                    temp_lst = copy.deepcopy(full_s_lst)
                    for idx, _ in enumerate(temp_lst):
                        temp_lst[idx] = temp_lst[idx] + '0'
                    for idx, _ in enumerate(full_s_lst):
                        full_s_lst[idx] = full_s_lst[idx] + '1'
                    full_s_lst = temp_lst + full_s_lst
            else:
                temp_lst = copy.deepcopy(full_s_lst)
                for idx, _ in enumerate(temp_lst):
                    temp_lst[idx] = temp_lst[idx] + '0'
                for idx, _ in enumerate(full_s_lst):
                    full_s_lst[idx] = full_s_lst[idx] + '1'
                full_s_lst = temp_lst + full_s_lst
        return full_s_lst

    def _extract_raw_reward(self, state, action):
        '''
        Get reward from the factored reward tree.
        '''
        state_dict = {}
        for idx, svar in enumerate(self.state_vars):
            if (state[idx] == '1'):
                state_dict[svar] = 'true'
            else:
                state_dict[svar] = 'false'
        if (action in self.valid_action_lst):
            a_ref = self.atomic_action_lst[self.valid_action_lst.index(action)]
            rwd = self.reward_table[a_ref]['extra']
            rwd_count = 1
            for case in self.reward_table[a_ref]['cases']:
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
                                    rwd_count += 1
                                    finished_search = True
                                break
                        if (updated):
                            updated = False
                        else:
                            finished_search = True
                if (count > maxiter):
                    raise Exception("reward tree is deeper than the number of state variables")
        else:
            raise Exception("Uncaptured case in extract raw reward")
        return rwd

    def _normalize_reward(self):
        '''
        Get the upper bound and lower bound of reward variable for normalizing reward into probabilities.
        '''
        self.r_max = max(self.r_max_lst)
        self.r_min = min(self.r_min_lst)
        self.r_min_unfactored = min(self.r_min_lst_unfactored)
        self.r_max_unfactored = max(self.r_max_lst_unfactored)
        if (self.r_min < 0):
            self.r_min = 3*self.r_min
        else:
            self.r_min = -3*self.r_min

    def _recover_full_dependency(self, s):
        '''
        Generate a list of states containing the partial irrelevant states grounding
        '''
        full_s_lst = ['']
        for idx, sv in enumerate(s):
            if (sv != '2'):
                for idx, _ in enumerate(full_s_lst):
                    full_s_lst[idx] = full_s_lst[idx] + sv
            else:
                # duplicate state list and take into account both '0' and '1' of the irrelevant var
                temp_lst = copy.deepcopy(full_s_lst)
                for idx, _ in enumerate(temp_lst):
                    temp_lst[idx] = temp_lst[idx] + '0'
                for idx, _ in enumerate(full_s_lst):
                    full_s_lst[idx] = full_s_lst[idx] + '1'
                full_s_lst = full_s_lst + temp_lst
        return full_s_lst

    def _enumerate_space(self, num_vars):
        '''
        generate the complete space from random variables
        '''
        space_enum = list(itertools.product(['0', '1'], repeat=num_vars))
        space = [''.join(g) for g in space_enum]
        return space

    def _init_vec_trans(self):
        '''
        Initialize vectorized transition matrix of dimension #state_space x #action_space
        where state space consists of all parent state variables and the child random variable as the last entry
        '''
        for child_s in self.state_dependency.keys():
            s_var = child_s.replace("'", "")
            self.vec_trans[s_var] = {}

            self.vec_trans[s_var]['parents'] = copy.deepcopy(self.state_dependency[child_s])
            self.vec_trans[s_var]['space'] = self._enumerate_space(len(self.vec_trans[s_var]['parents']) + 1)
            s_dim = len(self.vec_trans[s_var]['space'])
            self.vec_trans[s_var]['actions'] = self.atomic_action_lst
            a_dim = len(self.vec_trans[s_var]['actions'])
            self.vec_trans[s_var]['trans'] = -np.ones((s_dim, a_dim))

    def _generate_pseudo_trans(self):
        '''
        An enumeration version of the transition table.
        '''
        for child_s in self.state_dependency.keys():

            # generate full parent dependency table
            full_s_p_space = self._generate_full_space(len(self.state_dependency[child_s]))
            self.pseudo_trans[child_s] = {}
            for s_p in full_s_p_space:
                self.pseudo_trans[child_s][s_p] = {}
            for s, a, child_prob in zip(self.parent_s_table[child_s], self.parent_a_table[child_s], self.child_prob_table[child_s]):

                # recover full state dependency
                full_s_lst = self._recover_full_dependency(s)
                for f_s in full_s_lst:
                    self.pseudo_trans[child_s][f_s][a] = child_prob

    def _extract_compact_transition(self):
        '''
        Fill in
          - self.parent_s_table
          - self.parent_a_table
          - self.child_prob_table
          - self.parent_a_ref
        '''
        for child_s in self.state_dependency.keys():
            self.parent_s_table[child_s] = []
            self.parent_a_table[child_s] = []
            self.parent_a_ref[child_s] = []
            self.child_prob_table[child_s] = []
            self.compact_trans_dict[child_s] = []
            self.noop_table[child_s] = {}
            curr_s = child_s.replace("'", "")

            # Initialize the state value to be 2, for later easy-capturing vacuous state variable in some cases
            for act in self.trans_dict.keys():
                noop_status = False
                parent_a_val = ['0'] * len(self.action_vars)
                factored_act = act.split('___')
                for f_act in factored_act:
                        if (not f_act == 'noop'):
                            parent_a_val[self.action_vars.index(f_act)] = '1'
                        else:
                            noop_status = True
                for item in self.enum_table[act][curr_s]:

                    full_case = []
                    parent_s_val = ['2'] * len(self.state_dependency[child_s])
                    if (len(item[0]) != len(item[1])):
                        raise Exception("parent var not equal to parent val", item)
                    elif (len(item[0]) != 0):
                        for idx, var in enumerate(item[0]):
                            parent_s_val[self.state_dependency[child_s].index(var)] = item[1][idx]

                    self.parent_s_table[child_s].append(''.join(parent_s_val))
                    self.parent_a_table[child_s].append(''.join(parent_a_val))
                    full_case.append(''.join(parent_s_val))
                    full_case.append(''.join(parent_a_val))

                    if (parent_a_val.count("1") == 0):
                        self.parent_a_ref[child_s].append('noop')
                        full_case.append('noop')
                    elif (parent_a_val.count("1") == 1):
                        self.parent_a_ref[child_s].append(self.action_vars[parent_a_val.index("1")])
                        full_case.append(self.action_vars[parent_a_val.index("1")])
                    else:
                        a_lst = []
                        for idx, v in enumerate(parent_a_val):
                            if (v == '1'):
                                a_lst.append(self.action_vars[idx])
                        comb_act = self._find_atomic_action(a_lst)
                        self.parent_a_ref[child_s].append(comb_act)
                        full_case.append('___'.join(comb_act))

                    # generate vectorized transition dynamics
                    # if a state variable only depends on action, make it dependent on itself with invarient probability
                    full_s_lst = self._recover_full_dependency(''.join(parent_s_val))
                    a_idx = self.vec_trans[curr_s]['actions'].index(act)
                    for full_s in full_s_lst:
                        # append the probability of the child variable to be 0
                        s_idx = self.vec_trans[curr_s]['space'].index(full_s + '0')

                        self.vec_trans[curr_s]['trans'][s_idx, a_idx] = item[2][0]
                        # append the probability of the child variable to be 1
                        s_idx = self.vec_trans[curr_s]['space'].index(full_s + '1')
                        self.vec_trans[curr_s]['trans'][s_idx, a_idx] = item[2][1]

                    if (noop_status):
                        full_s_lst = self._recover_full_dependency(''.join(parent_s_val))
                        for f_s in full_s_lst:
                            self.noop_table[child_s][f_s] = item[2]

                    self.child_prob_table[child_s].append(item[2])
                    full_case.append(item[2])
                    self.compact_trans_dict[child_s].append(full_case)
            if (np.any(self.vec_trans[curr_s]['trans']) < 0):
                raise Exception('Vectorized transition  is incomplete, current vectorization is \n {}'.format(self.vec_trans[curr_s]['trans']))

    def extract_reward(self, state, action):
        '''
        return normalized reward as probability
        '''
        raw_rwd = self._extract_raw_reward(state, action)
        rwd = self._soften((raw_rwd - (self.r_min - self.epsilon))/(self.r_max - (self.r_min - self.epsilon)))
        return rwd

    def extract_exp_reward(self, state, action):
        '''
        return exponentiated reward as probability
        '''
        raw_rwd = self._extract_raw_reward(state, action)
        rwd = math.exp(raw_rwd)
        return rwd

    def generate_joint_table(self):
        '''
        Generate vectorized transition and reward for vectorized algorithmic usage.
        '''
        case_idx = -1
        for child in self.vec_trans.keys():
            case = {}
            case_idx += 1
            vars = copy.deepcopy(self.vec_trans[child]['parents'])
            vars.append(child+"'")
            case['row_var'] = vars
            case['row_val'] = self.vec_trans[child]['space']
            case['column_val'] = self.vec_trans[child]['actions']
            case['table'] = self.vec_trans[child]['trans']
            self.joint_trans_table.append(case)

            for var in case['row_var']:
                if (var not in self.trans_group.keys()):
                    self.trans_group[var] = [case_idx]
                else:
                    self.trans_group[var].append(case_idx)

        case_idx = -1
        for senario, raw_senario in zip(self.vec_rwds, self.vec_raw_rwds):
            case = {}
            raw_case = {}
            vars = []
            case_idx += 1
            vars = copy.deepcopy(senario['state_vars'])
            vars.append(senario['reward'])
            case['row_var'] = vars
            case['row_val'] = senario['space']
            case['column_val'] = senario['actions']
            case['table'] = senario['val']

            raw_case['row_var'] = vars
            raw_case['row_val'] = raw_senario['space']
            raw_case['column_val'] = raw_senario['actions']
            raw_case['table'] = raw_senario['val']
            
            self.joint_rwd_table.append(case)
            self.joint_raw_rwd_table.append(raw_case)

            for var in vars:
                if (var not in self.rwd_group.keys()):
                    self.rwd_group[var] = [case_idx]
                else:
                    self.rwd_group[var].append(case_idx)
            if (len(case['row_var']) == 1 and case['row_var'][0] == 'pr1'):
                self.rwd_group['action'] = [case_idx]

        exp_case_idx = -1
        for exp_senario in self.exp_vec_rwds:
            exp_case = {}
            exp_case_idx += 1
            exp_vars = copy.deepcopy(exp_senario['state_vars'])
            exp_vars.append(exp_senario['reward'])
            exp_case['row_var'] = exp_vars
            exp_case['row_val'] = exp_senario['space']
            exp_case['column_val'] = exp_senario['actions']
            exp_case['table'] = exp_senario['val']
            self.joint_exp_rwd_table.append(exp_case)

    def get_joint_trans(self):
        return self.joint_trans_table

    def get_joint_rwd(self):
        return self.joint_rwd_table

    def get_joint_raw_rwd(self):
        return self.joint_raw_rwd_table

    def get_joint_exp_rwd(self):
        return self.joint_exp_rwd_table

    def get_joint_exp_rwd(self):
        return self.joint_exp_rwd_table

    def get_trans_group(self):
        return self.trans_group

    def get_rwd_group(self):
        return self.rwd_group

    def get_vec_trans(self):
        return self.vec_trans

    def get_vec_rwds(self):
        return self.vec_rwds

    def get_exp_vec_rwds(self):
        return self.exp_vec_rwds

    def get_trans(self):
        return self.trans_dict

    def get_enum_table(self):
        return self.enum_table

    def get_pseudo_trans(self):
        return self.pseudo_trans

    def get_noop_table(self):
        return self.noop_table

    def get_r_max_lst(self):
        return self.r_max_lst

    def get_r_min_lst(self):
        return self.r_min_lst

    def get_init_state(self):
        return self.init_state

    def get_state_vars(self):
        return self.state_vars

    def get_action_vars(self):
        return self.action_vars

    def get_atomic_action_lst(self):
        return self.atomic_action_lst

    def get_valid_action_lst(self):
        return self.valid_action_lst

    def get_valid_action_pairs(self):
        return self.valid_action_pairs

    def get_reward_cases(self):
        return self.rwd_cases

    def get_reward_table(self):
        return self.reward_table

    def get_state_dependency(self):
        return self.state_dependency

    def get_parent_s_table(self):
        return self.parent_s_table

    def get_parent_a_table(self):
        return self.parent_a_table

    def get_parent_a_ref(self):
        return self.parent_a_ref

    def get_child_prob_table(self):
        return self.child_prob_table

    def determine_init_state(self):
        init = {}
        for svar in self.init_state.keys():
            val = self.init_state[svar].index(max(self.init_state[svar]))
            init[svar] = str(val)
        return init

    def get_compact_trans_dict(self):
        return self.compact_trans_dict

    def get_horizon(self):
        return self.horizon

    def get_discount(self):
        return self.discount

    def get_normal_factor(self):
        return self.r_min, self.r_max, self.epsilon

    def get_unfactored_bound(self):
        return self.r_min_unfactored, self.r_max_unfactored

    def get_domain(self):
        return self.domain

    def get_inst_name(self):
        return self.name

    def get_problem_size(self):
        max = 0
        min = 1e5
        sum = 0
        count = 0
        for var in self.state_dependency.keys():
            count += 1
            length = len(self.state_dependency[var])
            if (length > max):
                max = length
            if (length < min):
                min = length
            sum += length
        ave = sum/count
        act_size = len(self.atomic_action_lst)

        enum_max = 0
        enum_min = 1e5
        enum_sum = 0
        enum_count = 0
        for var in self.parent_s_table.keys():
            enum_count += 1
            length = len(self.parent_s_table[var])
            if (length > enum_max):
                enum_max = length
            if (length < enum_min):
                enum_min = length
            enum_sum += length
        enum_ave = enum_sum/enum_count

        act_count = 0
        for _ in self.reward_table.keys():
            act_count += 1
        if (act_size!= act_count):
            raise Exception("action error, {} action in reward table, {} action in atomic action list".format(act_count, act_size))
        return max, min, ave, act_size, enum_max, enum_min, enum_ave

    def print_reward(self):
        print("\nReward")
        for key3 in self.reward_table.keys():
            print(key3)
            print("Extra", self.reward_table[key3]['extra'])
            print("Action grounding", self.reward_table[key3]['value'])
            for item in self.reward_table[key3]['cases']:
                print(item)

    def print_trans(self):
        print("\nTransition")
        for key1 in self.trans_dict.keys():
            print(key1)
            for key2 in self.trans_dict[key1].keys():
                print(key2)
                print(self.trans_dict[key1][key2])

    def test(self):
        print("\nNormalize reward")
        print(self.r_max_lst)
        print(self.r_min_lst)
        print(self.get_normal_factor())
        print("\nInitialization")
        for key0 in self.init_state.keys():
            print(key0, self.init_state[key0])
        print("\ndiscount", self.discount)
        print("\nhorizon", self.horizon)

        print("\nState Dependency")
        for key4 in self.state_dependency.keys():
            print(key4)
            print("Depends on", self.state_dependency[key4])

        print("\nActions")
        print(self.action_vars)

        print("\nBruce Force")
        print(self.enum_table)

        print("\ntable")
        for s_var in self.state_dependency.keys():
            print("For state variable {}".format(s_var))
            print("\n","it is depend on states {} and actions {}".format(self.state_dependency[s_var], self.action_vars))
            for s, a, r, p in zip(self.parent_s_table[s_var], self.parent_a_table[s_var], self.parent_a_ref[s_var], self.child_prob_table[s_var]):
                print("\n", "For the grounding of state {} and action {},  (i.e. taking action {})".format(s, a, r))
                print("\n", "The distribution is {}".format(p))


if __name__ == "__main__":
    file = '../spudd_sperseus/skill_teaching_inst_mdp__1.spudd'
    parsed_ins = SPUDD_Parser(file)
    # print('vectorized transition dynamic \n')
    # print(parsed_ins.get_vec_trans())
    # # print('original transition dynamic \n')
    # # print(parsed_ins.get_trans())
    # print('vectorized reward table \n')
    # print(parsed_ins.get_vec_rwds())
    # # print('original reward table \n')
    # # print(parsed_ins.get_reward_table())
    # print('vectorized exponentiated reward table \n')
    # print(parsed_ins.get_exp_vec_rwds())
    # print("joint transition table")
    # print(parsed_ins.get_joint_trans())
    # print(parsed_ins.get_exp_vec_rwds())
    print("joint reward table")
    print(parsed_ins.get_joint_exp_rwd())
    print("joint exp reward table")
    print(parsed_ins.get_joint_rwd())
    # print("transition grouping")
    # print(parsed_ins.get_trans_group())
    # print("reward grouping")
    # print(parsed_ins.get_rwd_group())
