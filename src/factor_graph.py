#################################################################################
# Reference: https://github.com/krashkov/Belief-Propagation
#################################################################################

import numpy as np
import igraph as ig
import pyvis.network as net


class factor:

    def __init__(self, variables = None, distribution = None):
        if (distribution is None) and (variables is not None):
            self.__set_data(np.array(variables), None, None)
        elif (variables is None) or (len(variables) != len(distribution.shape)):
            raise Exception('Data is incorrect')
        else:
            self.__set_data(np.array(variables),
                            np.array(distribution),
                            np.array(distribution.shape))

    def __set_data(self, variables, distribution, shape):
        self.__variables    = variables
        self.__distribution = distribution
        self.__shape        = shape

    # ----------------------- Info --------------------------
    def is_none(self):
        return True if self.__distribution is None else False

    # ----------------------- Getters -----------------------
    def get_variables(self):
        return self.__variables

    def get_distribution(self):
        return self.__distribution

    def get_shape(self):
        return self.__shape


def factor_product(x, y):
    if x.is_none() or y.is_none():
        raise Exception('One of the factors is None')

    xy, xy_in_x_ind, xy_in_y_ind = np.intersect1d(x.get_variables(), y.get_variables(), return_indices=True)

    if xy.size == 0:
        raise Exception('Factors do not have common variables')

    if not np.all(x.get_shape()[xy_in_x_ind] == y.get_shape()[xy_in_y_ind]):
        raise Exception('Common variables have different order')

    x_not_in_y = np.setdiff1d(x.get_variables(), y.get_variables(), assume_unique=True)
    y_not_in_x = np.setdiff1d(y.get_variables(), x.get_variables(), assume_unique=True)

    x_mask = np.isin(x.get_variables(), xy, invert=True)
    y_mask = np.isin(y.get_variables(), xy, invert=True)

    x_ind = np.array([-1]*len(x.get_variables()), dtype=int)
    y_ind = np.array([-1]*len(y.get_variables()), dtype=int)

    x_ind[x_mask] = np.arange(np.sum(x_mask))
    y_ind[y_mask] = np.arange(np.sum(y_mask)) + np.sum(np.invert(y_mask))

    x_ind[xy_in_x_ind] = np.arange(len(xy)) + np.sum(x_mask)
    y_ind[xy_in_y_ind] = np.arange(len(xy))

    x_distribution = np.moveaxis(x.get_distribution(), range(len(x_ind)), x_ind)
    y_distribution = np.moveaxis(y.get_distribution(), range(len(y_ind)), y_ind)

    res_distribution =   x_distribution[tuple([slice(None)]*len(x.get_variables())+[None]*len(y_not_in_x))] \
                       * y_distribution[tuple([None]*len(x_not_in_y)+[slice(None)])]

    return factor(list(x_not_in_y)+list(xy)+list(y_not_in_x), res_distribution)


def factor_marginalization(x, variables):

    variables = np.array(variables)

    if x.is_none():
        raise Exception('Factor is None')

    if not np.all(np.in1d(variables, x.get_variables())):
        raise Exception('Factor do not contain given variables')

    res_variables    = np.setdiff1d(x.get_variables(), variables, assume_unique=True)
    res_distribution = np.sum(x.get_distribution(),
                              tuple(np.where(np.isin(x.get_variables(), variables))[0]))

    return factor(res_variables, res_distribution)


def factor_reduction(x, variable, value):
    if x.is_none() or (variable is None) or (value is None):
        raise Exception('Input is None')

    if not np.any(variable == x.get_variables()):
        raise Exception('Factor do not contain given variable')

    if value >= x.get_shape()[np.where(variable==x.get_variables())[0]]:
        raise Exception('Incorrect value of given variable')

    res_variables    = np.setdiff1d(x.get_variables(), variable, assume_unique=True)
    res_distribution = np.take(x.get_distribution(),
                               value,
                               int(np.where(variable==x.get_variables())[0]))

    return factor(res_variables, res_distribution)


def joint_distribution(ar):
    for element in ar:
        if element.is_none():
            raise Exception('Factor is None')

    res = ar[0]
    for element in ar[1:]:
        res = factor_product(res, element)

    return res


class factor_graph:
    def __init__(self):
        self._graph = ig.Graph()

    # ----------------------- Factor node functions ---------
    def add_factor_node(self, f_name, factor_): pass
    def change_factor_distribution(self, f_name, factor_): pass
    def remove_factor(self, f_name, remove_zero_degree=False): pass
    def __create_factor_node(self, f_name, factor_): pass

    # ----------------------- Rank functions -------
    def __check_variable_ranks(self, f_name, factor_, allowded_v_degree): pass
    def __set_variable_ranks(self, f_name, factor_): pass

    # ----------------------- Variable node functions -------
    def add_variable_node(self, v_name): pass
    def remove_variable(self, v_name): pass
    def __create_variable_node(self, v_name, rank=None): pass

    # ----------------------- Info --------------------------
    def get_node_status(self, name): pass

    # ----------------------- Graph structure ---------------
    def get_graph(self): pass
    def is_connected(self): pass
    def is_loop(self): pass


def change_factor_distribution(self, f_name, factor_):
    if self.get_node_status(f_name) != 'factor':
        raise Exception('Invalid variable name')
    if set(factor_.get_variables()) != set(self._graph.vs[self._graph.neighbors(f_name)]['name']):
        raise Exception('invalid factor distribution')

    # Check ranks
    self.__check_variable_ranks(f_name, factor_, 0)
    # Set ranks
    self.__set_variable_ranks(f_name, factor_)
    # Set data
    self._graph.vs.find(name=f_name)['factor_'] = factor_

factor_graph.change_factor_distribution = change_factor_distribution


def remove_factor(self, f_name, remove_zero_degree=False):
    if self.get_node_status(f_name) != 'factor':
        raise Exception('Invalid variable name')

    neighbors = self._graph.neighbors(f_name, mode="out")
    self._graph.delete_vertices(f_name)

    if remove_zero_degree:
        for v_name in neighbors:
            if self._graph.vs.find(v_name).degree() == 0:
                self.remove_variable(v_name)

factor_graph.remove_factor = remove_factor


def __create_factor_node(self, f_name, factor_):
    # Create node
    self._graph.add_vertex(f_name)
    self._graph.vs.find(name=f_name)['is_factor'] = True
    self._graph.vs.find(name=f_name)['factor_']   = factor_

    # Create corresponding edges
    start = self._graph.vs.find(name=f_name).index
    edge_list = [tuple([start, self._graph.vs.find(name=i).index]) for i in factor_.get_variables()]
    self._graph.add_edges(edge_list)

factor_graph.__create_factor_node = __create_factor_node


def __check_variable_ranks(self, f_name, factor_, allowded_v_degree):
    for counter, v_name in enumerate(factor_.get_variables()):
        if (self.get_node_status(v_name) == 'variable') and (not factor_.is_none()):
            if     (self._graph.vs.find(name=v_name)['rank'] != factor_.get_shape()[counter]) \
               and (self._graph.vs.find(name=v_name)['rank'] != None) \
               and (self._graph.vs.find(v_name).degree() > allowded_v_degree):
                raise Exception('Invalid shape of factor_')

factor_graph.__check_variable_ranks = __check_variable_ranks


def __set_variable_ranks(self, f_name, factor_):
    for counter, v_name in enumerate(factor_.get_variables()):
        if factor_.is_none():
            self._graph.vs.find(name=v_name)['rank'] = None
        else:
            self._graph.vs.find(name=v_name)['rank'] = factor_.get_shape()[counter]

factor_graph.__set_variable_ranks = __set_variable_ranks


def add_variable_node(self, v_name):
    if self.get_node_status(v_name) != False:
        raise Exception('Node already exists')
    self.__create_variable_node(v_name)

factor_graph.add_variable_node = add_variable_node


def remove_variable(self, v_name):
    if self.get_node_status(v_name) != 'variable':
        raise Exception('Invalid variable name')
    if self._graph.vs.find(v_name).degree() != 0:
        raise Exception('Can not delete variables with degree >0')
    self._graph.delete_vertices(self._graph.vs.find(v_name).index)

factor_graph.remove_variable = remove_variable


def __create_variable_node(self, v_name, rank=None):
    self._graph.add_vertex(v_name)
    self._graph.vs.find(name=v_name)['is_factor'] = False
    self._graph.vs.find(name=v_name)['rank'] = rank

factor_graph.__create_variable_node = __create_variable_node


def add_factor_node(self, f_name, factor_):
    if (self.get_node_status(f_name) != False) or (f_name in factor_.get_variables()):
        raise Exception('Invalid factor name')
    if type(factor_) is not factor:
        raise Exception('Invalid factor_')
    for v_name in factor_.get_variables():
        if self.get_node_status(v_name) == 'factor':
            raise Exception('Invalid factor')

    # Check ranks
    self.__check_variable_ranks(f_name, factor_, 1)
    # Create variables
    for v_name in factor_.get_variables():
        if self.get_node_status(v_name) == False:
            self.__create_variable_node(v_name)
    # Set ranks
    self.__set_variable_ranks(f_name, factor_)
    # Add node and corresponding edges
    self.__create_factor_node(f_name, factor_)

factor_graph.add_factor_node = add_factor_node


def get_node_status(self, name):
    if len(self._graph.vs) == 0:
        return False
    elif len(self._graph.vs.select(name_eq=name)) == 0:
        return False
    else:
        if self._graph.vs.find(name=name)['is_factor'] == True:
            return 'factor'
        else:
            return 'variable'

factor_graph.get_node_status = get_node_status


def get_graph(self):
    return self._graph

factor_graph.get_graph = get_graph


def is_connected(self):
    return self._graph.is_connected()

factor_graph.is_connected = is_connected


def is_loop(self):
    return any(self._graph.is_loop())

factor_graph.is_loop = is_loop


def string2factor_graph(str_):
    res_factor_graph = factor_graph()

    str_ = [i.split('(') for i in str_.split(')') if i != '']
    for i in range(len(str_)):
        str_[i][1] = str_[i][1].split(',')

    for i in str_:
        res_factor_graph.add_factor_node(i[0], factor(i[1]))

    return res_factor_graph


def plot_factor_graph(x, file_loc):
    graph = net.Network(notebook=True, width="100%")
    graph.toggle_physics(False)

    # Vertices
    label = x.get_graph().vs['name']
    color = ['#2E2E2E' if i is True else '#F2F2F2' for i in x.get_graph().vs['is_factor']]
    graph.add_nodes(range(len(x.get_graph().vs)), label=label, color=color)

    # Edges
    graph.add_edges(x.get_graph().get_edgelist())

    return graph.show(file_loc)
