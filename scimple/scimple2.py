"""
SCIMPLE, Parse and Plot scimply in 2 lines
Maintainer: enzobonnal@gmail.com
"""
import copy
import gc;

gc.collect()
import inspect
import multiprocessing
import os
import random
from collections import Collection
from threading import Thread

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # TODO : faire du pandas que si c'est  déjà install
from matplotlib import cm
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

_ = Axes3D


class ScimpleError(Exception):
    """module specific Errors"""


# #####
# TYPES
# #####

FuncType = type(lambda x: None)
NoneType = type(None)
inf = float('inf')

# #####
# TOOLS
# #####
def flatten_n_times(n, l):
    n = int(n)
    for _ in range(n):
        if any(type(elem) is list for elem in l):
            res = []
            for elem in l:
                res += elem if type(elem) is list else [elem]
            l = res
    return l

# #####
# PLOTS TOOLS
# #####
def xgrid(a, b, p):
    return list(flatten_n_times(1, [[i] * math.ceil((b - a) / p) for i in np.arange(a, b, p)]))


def ygrid(a, b, p):
    return list(flatten_n_times(1, [[i for i in np.arange(a, b, p)] for _ in np.arange(a, b, p)]))


# #####
# INFOS
# #####

def nb_params(f):
    """

    :param f: callable
    :return: number of parameters taken by f
    """
    return len(dict(inspect.signature(f).parameters))


# ######
# CHECKS
# ######

def type_value_checks(x, good_types=None, good_values=None, type_message='', value_message=''):
    """

    :param x: any
    :param good_types: type or Collection of types
    :param good_values: (callable : type(x) -> bool) or Collection of any
    :param type_message: str
    :param value_message:str
    :return: x type
    :raises: ValueError or TypeError
    """
    x_type = type(x)
    if good_types is not None:
        if isinstance(good_types, Collection):
            if not isinstance(x, tuple(good_types)):
                raise TypeError(type_message)
        else:
            if x_type != good_types:
                raise TypeError(type_message)
    if good_values is not None:
        if callable(good_values):
            if nb_params(good_values) != 1:
                raise ValueError("good_values must take exactly 1 argument")
            if not good_values(x):
                raise ValueError(value_message)
        elif isinstance(good_values, Collection):
            if x not in good_values:
                raise ValueError(value_message)
        else:
            raise ValueError("good_values must be either a callable or a Collection")
    return x_type


def is_2d_array_like_not_empty(array):
    """

    :param array: potential 2d array
    :return: bool
    """
    not_empty = False
    try:
        if len(array) == 0:
            return False
        for line in array:
            for _ in line:
                not_empty = True
    except TypeError:
        return False
    return not_empty


def is_color_code(color):
    """
    check if color is a valid color code like : '#457ef8'
    :param color: any
    :return: bool
    """
    if type(color) is not str:
        return False
    if len(color) != 7:
        return False
    if color[0] != '#':
        return False
    try:
        int('0x' + color[1:], 0)
        return True
    except ValueError:
        return False


# ######
# COLORS
# ######

def mult_array_mapped(a, f):
    """

    :param a: Collection
    :param f: function: elem of l --> float/int
    :return: product of the elements of a mapped by f
    """
    res = f(a[0])
    for i in range(1, len(a)):
        res *= f(a[i])
    return res


def random_color_well_dispatched(n):
    """

    :param n: number of color code well dispatched needed
    :return: list of n color codes well dispatched
    """
    pas = [1, 1, 1]
    while mult_array_mapped(pas, lambda x: x + 1) < n + 2 + (
            0 if pas[0] != pas[1] or pas[0] != pas[2] else pas[0] - 1):
        index_of_pas_min = pas.index(min(pas))
        pas[index_of_pas_min] += 1
    colors = []
    for r in range(pas[0] + 1):
        for v in range(pas[1] + 1):
            for b in range(pas[2] + 1):
                if not ((r == pas[0] and v == pas[1] and b == pas[2]) or (r == v == b == 0)):
                    if not (pas[0] == pas[1] == pas[2]) or not r == v == b:
                        colors.append([r, v, b])
    for i in range(len(colors)):
        color = colors[i]
        for j in range(3):
            color[j] = hex(int(255 / pas[j]) * color[j])[2:]
            if len(color[j]) == 1:
                color[j] = '0' + color[j]
        colors[i] = '#' + ''.join(color)
    return random.sample(colors, n)


def pastelize(color, coef_pastel=2, coef_fonce=0.75):
    """

    :param color: color code
    :param coef_pastel: merge les r v b
    :param coef_fonce: alpha coef
    :return: color code pastelized
    """
    if color[0] != '#':
        color = '#' + color
    colors = [int('0x' + color[1:3], 0), int('0x' + color[3:5], 0), int('0x' + color[5:7], 0)]
    for i in range(len(colors)):
        colors[i] = (colors[i] + (255 - colors[i]) / coef_pastel) * coef_fonce
    return '#' + ''.join(map(lambda x: hex(int(x))[2:], colors))


class Plot:
    """
    Plot object, associated with a unique figure
    """
    _gs = gridspec.GridSpec(2, 2, width_ratios=[50, 1], height_ratios=[1, 50])
    plt.rcParams['lines.color'] = 'b'

    _cm_column_index = 'column_index'
    _cm_column_name = 'column_name'
    _cm_color_code = 'color_code'
    _cm_function_float = 'function_float'
    _cm_function_color_code = 'function_color_code'
    _cm_function_str = 'function_str'

    def __init__(self, dim=2, title=None, xlabel=None, ylabel=None, zlabel=None, borders=None, bg_color=None):
        """

        :param dim: 2 or 3
        :param title: str
        :param xlabel: str
        :param ylabel: str
        :param zlabel: str, do not set if dim is not 3 !
        :param borders: Collection of int of length dim*2, settings axes limits
        :param bg_color: a valid color code str
        """
        self._at_least_one_label_defined = False
        self._color_bar_nb = 0
        self._fig = plt.figure()
        type_value_checks(dim, good_types=int, good_values={2, 3},
                          type_message='dim must be an integer',
                          value_message='dim value must be 2 or 3')
        if dim == 2:
            self._ax = self._fig.add_subplot(self._gs[2])
        elif dim == 3:
            self._ax = self._fig.add_subplot(self._gs[2], projection='3d')
        self._dim = dim
        if borders is not None:
            type_value_checks(borders, good_types={list, tuple}, good_values=lambda borders_: len(borders_) == dim * 2,
                              type_message="borders must be a list or a tuple",
                              value_message="length of borders list for 3D plot must be 6")
            self._ax.set_xlim(borders[0], borders[1])
            self._ax.set_ylim(borders[2], borders[3])
            if dim == 3:
                self._ax.set_zlim(borders[4], borders[5])
        if title is not None:
            type_value_checks(title, good_types=str, type_message="title type must be str")
            self._ax.set_title(title)
        if xlabel is not None:
            type_value_checks(xlabel, good_types=str, type_message="xlabel type must be str")
            self._ax.set_xlabel(xlabel)
        if ylabel is not None:
            type_value_checks(ylabel, good_types=str, type_message="ylabel type must be str")
            self._ax.set_ylabel(ylabel)
        if zlabel is not None:
            type_value_checks(zlabel, good_types=str, good_values=lambda zlabel_: dim != 3,
                              type_message="zlabel type must be str",
                              value_message="zlabel is only settable on 3D plots")
            self._ax.set_zlabel(zlabel)
        if bg_color is not None:
            type_value_checks(bg_color, good_types=str, good_values=is_color_code,
                              type_message="bg_color must be of type str",
                              value_message="bg_color_must be a valid hexadecimal color code, example : '#ff45ab'")
            self._ax.set_facecolor(bg_color)

    def _print_color_bar(self, mini, maxi, color_bar_label=None):
        """
        Called when colored_by of type function float is used : creates a 'legend subplot' describing the values
        Only 2 color bars can be plotted on a same plot
        :param mini: int/float
        :param maxi: int/float
        :param color_bar_label: str
        :return: None
        """
        self._color_bar_nb += 1
        if self._color_bar_nb == 3:
            raise ScimpleError("only 2 function-colored plots available")
        color_bar_subplot = self._fig.add_subplot(self._gs[3 if self._color_bar_nb == 1 else 0])
        color_bar_subplot.imshow(
            [[i] for i in np.arange(maxi, mini, (mini - maxi) / 100)] if self._color_bar_nb == 1 else
            [[i for i in np.arange(maxi, mini, (mini - maxi) / 100)]],
            interpolation='nearest',
            cmap=[cm.gray, cm.autumn][self._color_bar_nb - 1],
            extent=[maxi / 70, mini / 70, mini, maxi] if self._color_bar_nb == 1 else [maxi, mini, mini / 70,
                                                                                       maxi / 70])
        color_bar_subplot.ticklabel_format(axis='yx', style='sci', scilimits=(-2, 2))
        color_bar_subplot.legend().set_visible(False)
        if self._color_bar_nb == 1:
            color_bar_subplot.set_xticks([])
            if color_bar_label:
                color_bar_subplot.set_ylabel(color_bar_label)
            color_bar_subplot.yaxis.tick_right()
        else:
            color_bar_subplot.set_yticks([])
            color_bar_subplot.xaxis.set_label_position('top')
            if color_bar_label:
                color_bar_subplot.set_xlabel(color_bar_label)
            color_bar_subplot.xaxis.tick_top()

    @staticmethod
    def _coloring_mode(colored_by, test_index, xyz_tuple):
        """

        :param colored_by: argument given to Plot.add()
        :return: str describing the mode, one of the following :
                'column_index', 'column_name', 'color_code', 'function_float', 'function_color_code'
                'function_str' or None if no mode match
        """
        if type(colored_by) is int:
            return Plot._cm_column_index
        if is_color_code(colored_by):
            return Plot._cm_color_code
        if type(colored_by) is str:
            return Plot._cm_column_name
        if callable(colored_by):
            if nb_params(colored_by) != 2:
                return None
            res = colored_by(test_index, xyz_tuple)
            try:
                float(res)
                return Plot._cm_function_float
            except ValueError:
                pass
            if is_color_code(res):
                return Plot._cm_function_color_code
            if type(res) is str:
                return Plot._cm_function_str
        return None

    def add(self, table=None, x=None, y=None, z=None, first_line=0, last_line=None,
            label=None, colored_by=None, color_bar_label=None, marker='-', markersize=9):
        """

        :param table:
            None
            scimple.Table
        :param x:
            int : table column index
            str : table column name
            collections.Collection : 1D array-like
        :param y:
            int : table column index
            str : table column name
            collections.Collection : 1D array-like
            function type(x_plot element) x, int index -> int/str
        :param z: Should be set if plot dimension is 3 and kept as None if dim != 3
            None
            int : table column index
            str : table column name
            collections.Collection : 1D array-like
            function type(x_plot element) x, type(y_plot element) y,int index -> int/str
        :param first_line:
            int : first row to be considered in table (indexing starts at 0)
        :param last_line:
            None
            int : last row (included) to be considered in table (indexing starts at 0)
        :param label:
            None
            str : label of the plot
            dict : {color_code color : str label}
        :param colored_by:
            None
            str : describing the mode, one of the following :
                int : column_index integer
                color code
                str : column_name
                function int : index, tuple : xyz_tuple ->float/int
                function int : index, tuple : xyz_tuple ->(color_code, str_label)
                function int : index, tuple : xyz_tuple ->str_class
        :param marker:
            str : single character (matplotlib marker)
        :param markersize:
            int
        :return:
        """
        # first_line
        type_value_checks(first_line, good_types=int,
                          type_message="first_line must be an integer",
                          good_values=lambda first_line: first_line >= 0,
                          value_message="first_line should be positive")
        # last_line
        if last_line is None and table is not None:
            last_line = len(table) - 1
        type_value_checks(last_line, good_types=(int, NoneType),
                          type_message="last_line must be an integer",
                          good_values=lambda last_line: type(last_line) is NoneType or last_line >= first_line,
                          value_message="last_line should be greater than or equal to first_line (default 0)")
        # variables:
        x_y_z_collections_len = last_line - first_line + 1 if last_line else None
        kwargs_plot = {}
        # table
        type_value_checks(table, good_types=(Table, NoneType),
                          type_message='table must be an instance of class scimple.Table or None')
        # x
        x_type = type_value_checks(x, good_types=(Collection, int, str),
                                   type_message="x must be a Collection (to plot) or (only if table set)" +
                                                "an int (table column index) or a str (table column name)",
                                   good_values=lambda x:
                                   0 <= x < len(table.columns) if table is not None and type(x) is int else
                                   False if table is None and type(x) is int else
                                   x in table.columns if table is not None and type(x) is str else
                                   len(x) == x_y_z_collections_len if table is not None else
                                   len(x) > 0,
                                   value_message="x value must verify : 0 <= x < len(table.columns)"
                                   if table is not None and type(x) is int else
                                   "x can't be integer if table not set"
                                   if table is None and type(x) is int else
                                   "x must be in table.columns" if table is not None and type(x) is str else
                                   "x must be a collection " +
                                   "verifying : len(x) == last_line-first_line+1" if table is not None else
                                   "x must be a collection " +
                                   "verifying : len(x) > 0")
        if issubclass(x_type, Collection):
            x_y_z_collections_len = len(x)
        # y
        y_type = type_value_checks(y, good_types=(Collection, int, str, FuncType),
                                   type_message="y must be a Collection (to plot) or (only if table set)" +
                                                "an int (table column index) or a str (table column name)",
                                   good_values=lambda y:
                                   0 <= y < len(table.columns) if table is not None and type(y) is int else
                                   False if table is None and type(x) is int else
                                   y in table.columns if table is not None and type(y) is str else
                                   len(y) == x_y_z_collections_len if isinstance(y, Collection) else
                                   nb_params(y) == 2 if callable(y) else False,
                                   value_message="y value must verify : 0 <= y < len(table.columns)"
                                   if table is not None and type(y) is int else
                                   "y can't be integer if table not set"
                                   if table is None and type(y) is int else
                                   "y must be in table.columns" if table is not None and type(y) is str else
                                   "y must be a collection " +
                                   "verifying : len(y) == len(x)" if table is not None and isinstance(y,
                                                                                                      Collection) else
                                   "y must be a collection verifying : len(y) > 0" if isinstance(y, Collection) else
                                   "y function must take exactly 1 argument (type(x element))" if callable(y) else
                                   "y value invalid")

        # z
        z_type = type_value_checks(z, good_types=(Collection, int, str, FuncType, NoneType),
                                   type_message="z must be a Collection (to plot) or (only if table set)" +
                                                "an int (table column index) or a str (table column name) or None",
                                   good_values=lambda z:
                                   False if self._dim != 3 and z is not None else
                                   True if z is None else
                                   0 <= z < len(table.columns) if table is not None and type(z) is int else
                                   False if table is None and type(x) is int else
                                   z in table.columns if table is not None and type(z) is str else
                                   len(z) == x_y_z_collections_len if isinstance(z, Collection) else
                                   nb_params(z) == 3 if callable(z) else False,
                                   value_message="z is only settable in 3D plots"
                                   if self._dim != 3 and z is not None else
                                   "z value must verify : 0 <= z < len(table.columns)"
                                   if table is not None and type(z) is int else
                                   "z can't be integer if table not set"
                                   if table is None and type(z) is int else
                                   "z must be in table.columns" if table is not None and type(z) is str else
                                   "z must be a collection " +
                                   "verifying : len(z) == len(x)" if table is not None and isinstance(z,
                                                                                                      Collection) else
                                   "z must be a collection verifying : len(z) > 0" if isinstance(z, Collection) else
                                   "z function must take exactly 2 argument (type(x element,y element))" if callable(z)
                                   else "z value invalid")

        # label
        type_value_checks(label, good_types=(str, NoneType, dict),
                          type_message="label must be a string or a dict colorcode : label")

        # marker
        type_value_checks(marker, good_types=(str, NoneType),
                          type_message="marker must be a string",
                          good_values=lambda marker: not marker or len(marker) == 1,
                          value_message="marker can only be a matplotlib marker like 'o' or '-' (one character)")

        # markersize
        type_value_checks(markersize, good_types=int,
                          type_message="markersize must be an int",
                          good_values=lambda markersize: markersize >= 0,
                          value_message="marker musr be positive")
        kwargs_plot['markersize'] = markersize

        # arrays to plot
        z_plot = None
        if x_type in {int, str}:
            x_plot = table[first_line:last_line + 1, x]
        elif issubclass(x_type, Collection):
            if x_type is pd.Series:
                x_plot = x.as_matrix()
            else:
                x_plot = x
        else:
            raise Exception("should never happen 48648648")

        if y_type in {int, str}:
            y_plot = table[first_line:last_line + 1, y]
        elif issubclass(y_type, Collection):
            if y_type is pd.Series:
                y_plot = y.as_matrix()
            else:
                y_plot = y
        elif y_type is FuncType:
            y_plot = [y(i, x[i]) for i in range(x_y_z_collections_len)]
        else:
            raise Exception("should never happen 86789455")

        if z_type is NoneType:
            pass
        elif z_type in {int, str}:
            z_plot = table[first_line:last_line + 1, z]
        elif issubclass(z_type, Collection):
            if z_type is pd.Series:
                z_plot = z.as_matrix()
            else:
                z_plot = z
        elif z_type is FuncType:
            z_plot = [z(i, x[i], y[i]) for i in range(x_y_z_collections_len)]
        else:
            raise Exception("should never happen 78941153")

        if not (x_plot and y_plot):
            raise Exception("should never happen 448789")

        # to_plot (need to be before # colored_by )

        if self._dim == 3:  # 3D
            to_plot = (x_plot, y_plot, z_plot)
        else:  # 2D
            to_plot = (x_plot, y_plot)

        # colored_by:
        color_mode = self._coloring_mode(colored_by, first_line,
                                         [[to_plot[i][0]] for i in range(len(to_plot))])
        type_value_checks(color_mode, good_types=(str, NoneType),
                          type_message='colored_by is not a valid colored_by parameters, should be one of the' +
                                       'following :\n' + "'column_index integer', 'color_code'," +
                                       "'function int : index, tuple : xyz_tuple ->float/int'," +
                                       "'function int : index, tuple : xyz_tuple ->color_code'",
                          good_values=lambda color_mode:
                          False if color_mode is None and colored_by is not None else
                          False if color_mode == Plot._cm_column_index
                                   or color_mode == Plot._cm_column_name
                                   and table is None else True,
                          value_message="colored_by is not a valid mode"
                          if color_mode is None and colored_by is not None else
                          "colored_by can't be a column index/name if table parameter is not set")
        if color_mode == Plot._cm_column_name:  # from column name to index
            colored_by = table.get_column_index_from_name(colored_by)
            color_mode = Plot._cm_column_index
        # adding marker/fmt : (need to be after # color_mode )
        to_plot = (*to_plot, marker)
        # Plots following the color_mode
        if color_mode is None:
            if type(label) is str and len(label) != 0:  # label != from None and ""
                kwargs_plot['label'] = label
                self._at_least_one_label_defined = True
            self._ax.plot(*to_plot, **kwargs_plot)
        elif color_mode == Plot._cm_color_code:
            if type(label) is str and len(label) != 0:  # label != from None and ""
                kwargs_plot['label'] = label
                self._at_least_one_label_defined = True
            elif type(label) is dict and colored_by in label:
                kwargs_plot['label'] = label[colored_by]
                self._at_least_one_label_defined = True
            kwargs_plot['color'] = colored_by
            self._ax.plot(*to_plot, **kwargs_plot)
        elif color_mode in [Plot._cm_column_index, Plot._cm_function_str]:
            self._at_least_one_label_defined = True
            xyz = (x_plot, y_plot, z_plot) if self._dim == 3 else (x_plot, y_plot)
            # build dict_groups
            dict_groups = {}
            for index in range(len(x_plot)):
                group = (table[index][colored_by] if color_mode == Plot._cm_column_index
                         else colored_by(index, xyz))
                if group in dict_groups:
                    dict_groups[group] += [index]
                else:
                    dict_groups[group] = [index]
            colors_list = random_color_well_dispatched(len(dict_groups))
            for group in dict_groups:
                x_group, y_group, z_group = list(), list(), list()
                for index in dict_groups[group]:
                    x_group.append(x_plot[index])
                    y_group.append(y_plot[index])
                    if self._dim == 3:
                        z_group.append(z_plot[index])
                if self._dim == 2:
                    to_plot = (x_group, y_group)
                elif self._dim == 3:
                    to_plot = (x_group, y_group, z_group)
                self._ax.plot(*(*to_plot, marker),
                              **{**kwargs_plot, 'label':group , 'color': pastelize(colors_list.pop())})

        elif color_mode == Plot._cm_function_color_code:
            if type(label) is str and len(label) != 0:  # label != from None and ""
                kwargs_plot['label'] = label
                self._at_least_one_label_defined = True
            xyz = (x_plot, y_plot, z_plot) if self._dim == 3 else (x_plot, y_plot)
            dict_color_to_lines = dict()  # hexa -> plotable lines
            for index in range(len(x_plot)):
                color = colored_by(index, xyz)
                if color in dict_color_to_lines:
                    dict_color_to_lines[color][0] += [x_plot[index]]
                    dict_color_to_lines[color][1] += [y_plot[index]]
                    if self._dim == 3:
                        dict_color_to_lines[color][2] += [z_plot[index]]
                else:
                    dict_color_to_lines[color] = [[x_plot[index]],
                                                  [y_plot[index]]]
                    if self._dim == 3:
                        dict_color_to_lines[color] = [[x_plot[index]],
                                                      [y_plot[index]],
                                                      [z_plot[index]]]

            for color in dict_color_to_lines:
                if type(label) is dict and color in label:
                    kwargs_plot['label'] = label[color]
                    self._at_least_one_label_defined = True
                if self._dim == 2:
                    self._ax.plot(*(dict_color_to_lines[color][0], dict_color_to_lines[color][1], marker),
                                  **{**kwargs_plot, 'color': color, 'solid_capstyle': "round"})
                elif self._dim == 3:
                    self._ax.plot(
                        *(dict_color_to_lines[color][0], dict_color_to_lines[color][1],
                          dict_color_to_lines[color][2],
                          marker), **{**kwargs_plot, 'color': color, 'solid_capstyle': "round"})
        elif color_mode == Plot._cm_function_float:
            maxi = -inf
            mini = inf
            xyz = (x_plot, y_plot, z_plot) if self._dim == 3 else (x_plot, y_plot)
            for i in range(len(x_plot)):
                try:
                    value = colored_by(i, xyz)
                    maxi = max(maxi, value)
                    mini = min(mini, value)
                except:
                    raise ValueError('colored_by function failed on parameters :' +
                                     '\ni=' + str(i) +
                                     '\nxyz_tuple=' + str(xyz))
            self._print_color_bar(mini, maxi, label)
            dict_color_to_lines = {}  # hexa -> plotable lines
            for index in range(len(x_plot)):
                maxolo = max(0, colored_by(index, (x_plot, y_plot, z_plot)) - mini)
                minolo = min(255, maxolo * 255 / (maxi - mini))
                color_hexa_unit = hex(int(minolo))[2:]
                if len(color_hexa_unit) == 1:
                    color_hexa_unit = "0" + color_hexa_unit
                color = "#" + color_hexa_unit * 3 \
                    if self._color_bar_nb == 1 else \
                    "#ff" + color_hexa_unit + "00"
                if color in dict_color_to_lines:
                    dict_color_to_lines[color][0] += [x_plot[index]]
                    dict_color_to_lines[color][1] += [y_plot[index]]
                    if self._dim == 3:
                        dict_color_to_lines[color][2] += [z_plot[index]]
                else:
                    dict_color_to_lines[color] = [[x_plot[index]],
                                                  [y_plot[index]]]
                    if self._dim == 3:
                        dict_color_to_lines[color] = [[x_plot[index]],
                                                      [y_plot[index]],
                                                      [z_plot[index]]]
            if self._dim == 2:
                for color in dict_color_to_lines:
                    self._ax.plot(*(dict_color_to_lines[color][0], dict_color_to_lines[color][1], marker),
                                  **{**kwargs_plot, 'color': color, 'solid_capstyle': "round"})
            elif self._dim == 3:
                for color in dict_color_to_lines:
                    self._ax.plot(
                        *(dict_color_to_lines[color][0], dict_color_to_lines[color][1], dict_color_to_lines[color][2],
                          marker), **{**kwargs_plot, 'color': color, 'solid_capstyle': "round"})

        else:
            raise Exception("should never happen 4567884565")
        if self._at_least_one_label_defined:
            self._ax.legend(loc='upper right', shadow=True).draggable()
        return self


def show(block=True):
    plt.show(block=block)


class Table:
    def __init__(self, path, first_line=0, last_line=None, column_names=None, delimiter=r'(	|[ ])+',
                 new_line=r'(	| )*(()|)', float_dot='.', number_format_character='',
                 ignore="", print_tokens=False, print_error=False, header=None):

        # dev args:
        self._print_tokens = print_tokens
        self._print_error = print_error
        # init fields
        self._path = path  # string
        self._first_line = first_line  # int
        if (last_line is not None and last_line < 0):
            print("S_c_i_m_p_l_e E_r_r_o_r : last_line Argument must be >=1")
            raise Exception()
        self._last_line = last_line  # int
        self._content_as_string = ""  # string
        self._float_table = []  # string
        self._delimiter = delimiter
        self._new_line = new_line
        self._float_dot = (r'\.' if float_dot == '.' else float_dot)  # reg_exp
        self._number_format_character = number_format_character  # string
        self._ignore = ignore  # reg_exp
        # Map_reduce:
        self._mapping = None
        # import file
        if isinstance(path, Table):
            column_names = list(path.get_columns_names().keys())
            if first_line == 1 and last_line is None:
                self._float_table
            else:
                self._float_table = copy.deepcopy(path[max(0, first_line):last_line + 1] if last_line
                                                  else path[max(0, first_line):])
        elif type(path) is not str:
            try:
                self._float_table = [list(line)[:] for line in
                                     (path[max(0, first_line):last_line + 1] if last_line else path[
                                                                                               max(0, first_line):])]
            # print('input considered as array-like')
            except Exception:
                print('Unsupported array-like object in input')
                raise
        else:
            try:
                in_file = open(path, 'r')
                self._content_as_string = in_file.read()
                in_file.close()
                self._parse()
            # print('input considered as path to file')
            except OSError:
                self._content_as_string = path
                self._parse()
            # print('input considered as string content')
        if column_names is None or len(column_names) == 0:
            if header is not None:
                self._column_names = {str(self[header][i]): i for i in
                                      range(len(self[header]))}  # list
                del self[header]
            else:
                self._column_names = {}
        else:
            self._column_names = {column_names[i]: i for i in range(len(column_names))}  # list

    def __contains__(self, elem_to_find):
        for line in self.get_table():
            for elem in line:
                if elem_to_find == elem:
                    return True
        return False

    def __str__(self):
        return self.get_string()

    def __unicode__(self):
        return self.get_string()

    def __repr__(self):
        return str(self.get_table())

    def __iter__(self):
        return iter(self.get_table())

    def __bool__(self):
        return bool(len(self.get_table()))

    def _from_col_name_to_int(self, t):
        if type(t) == int:
            return t
        elif type(t) == str:
            return self._column_names[t]
        else:
            return tuple({(self._column_names[elem] if type(elem) is str else elem) for elem in t})

    def __getitem__(self, t):
        '''

        :param t: tuple/str : accès aux colonnes (nom ou numero (debut à 0), int : accès à une ligne, slice : accès à des lignes
        :return:
        '''
        if type(t) in {slice, int}:
            return self.get_table()[t]
        if type(t) is str:
            return self.get_table(t)
        if type(t) is tuple:
            t = self._from_col_name_to_int(t)
            return self.get_table(t)

    def __delitem__(self, t):
        if type(t) in {slice, int}:
            del self.get_table()[t]
        if type(t) is str:
            self._filter_by_columns_del_keep('del', t)
        if type(t) is tuple:
            self._filter_by_columns_del_keep('del', t)

    def __len__(self):
        return len(self.get_table())

    def append(self, elem):
        self.get_table().append(list(elem))
        return self

    def pop(self, columns_num_tuple, *columns_num):
        if type(columns_num_tuple) is str:
            columns = [columns_num_tuple]
            for name in columns_num:
                columns.append(name)
            columns = tuple(columns)
        else:
            columns = columns_num_tuple
        if type(columns) in {slice, int}:
            popped = self.get_table()[columns]
            del self.get_table()[columns]
        elif type(columns) is str:
            popped = self.get_table(columns)
            self._filter_by_columns_del_keep('del', columns)
        elif type(columns) is tuple:
            popped = self.get_table(columns)
            self._filter_by_columns_del_keep('del', columns)
        return popped

    def _parse(self):
        # List of token names.
        tokens = (
            'delimiter',
            'newLine',
            'char'
        )
        # variable :
        lineNumber = 0

        # Regles :

        def t_newLine(t):
            r''
            t.lexer.lineno += 1
            return t

        t_newLine.__doc__ = self._new_line

        def t_delimiter(t):
            r''
            return t

        t_delimiter.__doc__ = self._delimiter

        def t_char(t):
            r'.'
            return t

        t_ignore = self._ignore

        def t_eof(t):
            return t

        # en cas d'ERROR :
        def t_error(t):
            if self._printError:
                print("Error on char : '%s'" % t.value[0])  # dev
            t.lexer.skip(1)

        # Build du lexer
        lexer = lex()
        # On donne l'input au lexer
        lexer.input(self._contentAsString)
        # On build la string résultat :
        currentLine = list()
        currentChars = ''
        tok = lexer.token()
        last_tok = None
        while tok:
            if tok.lineno >= self._firstLine + 1 and (self._lastLine is None or tok.lineno <= self._lastLine + 1):
                if tok.type == "newLine":
                    currentLine.append(self._try_to_float(currentChars))
                    currentChars = ''
                    self._floatTable.append(currentLine)
                    currentLine = []
                elif tok.type == "delimiter":
                    currentLine.append(self._try_to_float(currentChars))
                    currentChars = ''
                else:
                    currentChars += tok.value
            elif not (self._lastLine is None or tok.lineno <= self._lastLine + 1):
                break
            if self._printTokens:
                print(tok)
            tok = lexer.token()
        if not tok and (self._lastLine is None or tok.lineno <= self._lastLine + 1):
            currentLine.append(self._try_to_float(currentChars))
            self._floatTable.append(currentLine)

    def _try_to_float(self, s):
        try:
            chars_copied = s
            chars_copied = chars_copied.replace(self._number_format_character, '')
            if self._float_dot != '\.':
                chars_copied = chars_copied.replace(self._float_dot, '.')
            chars_copied = float(chars_copied)
            if chars_copied % 1 == 0:
                chars_copied = int(chars_copied)
            return chars_copied
        except ValueError:
            return s

    # public :
    def get_columns_names(self):
        return self._column_names

    def set_columns_names(self, columns_num_tuple, *columns_num):
        if type(columns_num_tuple) is str:
            columns = [columns_num_tuple]
            for name in columns_num:
                columns.append(name)
        else:
            columns = columns_num_tuple
        self._column_names = {columns[i]: i for i in range(len(columns))}

    def keep_columns(self, columns_num_tuple, *columns_num):
        return self._filter_by_columns_del_keep('keep', columns_num_tuple, *columns_num)

    def _filter_by_columns_del_keep(self, del_or_keep, columns_num_tuple, *columns_num):
        if type(columns_num_tuple) in {str, int}:
            columns = [columns_num_tuple]
            for name in columns_num:
                columns.append(name)
        else:
            columns = columns_num_tuple
        columns = self._from_col_name_to_int(columns)
        for line in self:
            i = 0
            j = 0
            while i < len(line):
                if (del_or_keep == 'del' and i in columns) or (del_or_keep == 'keep' and i not in columns):
                    del line[j]
                    i += 1
                else:
                    j += 1
                    i += 1
        i = 0
        j = 0
        new_columns_names = dict()
        for key in self._column_names.keys():
            if (del_or_keep == 'del' and i not in columns) or (del_or_keep == 'keep' and i in columns):
                new_columns_names[key] = j
                j += 1
            i += 1

        self._column_names = new_columns_names

        return self

    def get_table(self, columns_num_tuple=None, *columns_num):
        '''returns the table (list of list) of floats with None for empty fields'''
        if columns_num_tuple is not None:
            if type(columns_num_tuple) in {str, int}:
                columns = [columns_num_tuple]
                for name in columns_num:
                    columns.append(name)
            else:
                columns = columns_num_tuple
            columns = self._from_col_name_to_int(columns)
            res = [[line[i] for i in range(len(line)) if i in columns] for line in self]
            return res
        else:
            return self._float_table

    def get_string(self, delimiter=None, new_line=None):
        if delimiter == None: delimiter = self._delimiter
        if new_line == None: new_line = self._new_line
        self._content_as_string = ""
        if delimiter == r'(	|[ ])+':
            delimiter = ','
        if new_line == r'(	| )*(()|)':
            new_line = ''
        new_line = new_line.replace("\n", "").replace("\t", "	")
        delimiter = delimiter.replace("\n", "").replace("\t", "	")

        self._content_as_string = new_line.join([delimiter.join([str(elem) for elem in line]) for line in self])

        return self._content_as_string

    def get_copy(self):
        return copy.deepcopy(self)

# #######
# CSV I/O
# #######
def save_csv(path, delimiter=',', separator='\n'):
    f = open(path, 'w')
    f.write(self.get_string(delimiter, new_line))
    f.close()

def load_csv(path, delimiter=None, new_line=None):
    f = open(path, 'r')
    f.write(self.get_string(delimiter, new_line))

    # Map_reduce :
    def _init_mapping(self):
        if self._mapping == None:
            self._mapping = dict()
            for line_num in range(len(self)):
                self._mapping[line_num + 1] = [self[line_num]]

    def reset_mapping(self):
        self._mapping = dict()
        for line_num in range(len(self)):
            self._mapping[line_num + 1] = [self[line_num]]
        return self

    def get_mapping(self):
        self._init_mapping()
        return self._mapping

    def get_mapping_as_table(self, flatten=False):

        return [[key, self.get_mapping()[key]] if not flatten else [key]
                                                                   + flatten_n_times(flatten - 1,
                                                                                     self.get_mapping()[key])
                for key in self.get_mapping()]

    def _build_mr_task(self, key, value_s, f, new_mapping):
        newpairs = f(key, value_s)
        if newpairs is not None:
            if type(newpairs) is list:
                newpairs = f(key, value_s)
                for newkey, newvalue in newpairs:
                    if newkey in new_mapping:
                        new_mapping[newkey].append(newvalue)
                    else:
                        new_mapping[newkey] = [newvalue]
            elif type(newpairs) is tuple:
                newkey, newvalue = f(key, value_s)
                if newkey in new_mapping:
                    new_mapping[newkey].append(newvalue)
                else:
                    new_mapping[newkey] = [newvalue]

    class _M_r_thread(Thread):
        def __init__(self, type, keys, f, new_mapping, table):
            Thread.__init__(self)
            self._type = type
            self._keys = keys
            self._f = f
            self._new_mapping = new_mapping
            self._table = table

        def run(self):
            if self._type == 'map':
                for key in self._keys:
                    for value in self._table._mapping[key]:
                        self._table._build_mr_task(key, value, self._f, self._new_mapping)
            elif self._type == 'reduce':
                for key in self._keys:
                    self._table._build_mr_task(key, self._table._mapping[key], self._f, self._new_mapping)
            else:
                print("error code 62786289629")

    def _melt_mappings(self, mappings):
        melted_mapping = dict()
        for mapping in mappings:
            for key in mapping:
                if key in melted_mapping:
                    melted_mapping[key] += mapping[key][:]
                else:
                    melted_mapping[key] = mapping[key][:]
        return melted_mapping

    def _perform_map_or_reduce(self, type, f, threads):
        new_mappings = [{} for _ in range(threads)]
        keys = list(self.get_mapping().keys())  # _init_mapping() ran during get_mapping
        threads = min(threads, len(keys))
        keys_parts = [[] for _ in range(threads)]
        for i in range(len(keys)):
            keys_parts[i % threads].append(keys[i])
        threads_list = list()
        for i in range(threads):
            threads_list.append(self._M_r_thread(type, keys_parts[i], f, new_mappings[i], self))
        for thread in threads_list:
            thread.start()
        for thread in threads_list:
            thread.join()
        self._mapping = self._melt_mappings(new_mappings)

    def map(self, f, threads=multiprocessing.cpu_count()):
        '''
        f du type lambda key value : return (key, value)
        :return: self (to chain)
        '''
        self._perform_map_or_reduce('map', f, threads)
        return self

    def reduce(self, f, threads=multiprocessing.cpu_count()):
        '''
        f : lambda key, values_list : key, value
        :param f:
        :return: self (to chain)
        '''
        self._perform_map_or_reduce('reduce', f, threads)
        return self



def _get_data(path):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scimple_data', path)


def get_sample(id):
    dic = {'xyz': "phenyl-Fe-porphyirine-C_o2-Me_4_rel.xyz",
           'charges': 'C_h_a_r_g_e_s_phenyl-Fe-porphyirine-C_o2-Me_4_rel',
           'surfaces': 'ek_In_t_p_C_o2_Me_4_graphene_W_r2_k.dat',
           'adults': 'adult.txt'}
    if id == 'xyz':
        return Table(Table(_get_data(dic[id]), column_names=['rien', 'atom', 'x', 'y', 'z'],
                           last_line=494)['atom', 'x', 'y', 'z'],
                     column_names=['atom', 'x', 'y', 'z'])
    elif id == 'charges':
        res = Table(Table(_get_data(dic[id]), header=1, last_line=494)['s', 'p', 'd'], column_names=['s', 'p', 'd'])
        return res
    elif id == 'surfaces':
        return Table(_get_data(dic[id]))
    elif id == 'adults':
        return Table(_get_data(dic[id]), header=0, delimiter=',', last_line=100)


def run_example():
    source = ''''''


if __name__ == '__main__':
    # run_example()
    tab = get_sample('xyz')
    print(tab)
    Plot(2, title=':)').add(x=range(100), y=lambda i, x: 50*math.sin(x/10),
                            marker='.', colored_by=lambda i, xy: xy[1][i], label='du noir au blanc') \
        .add(x=range(100), y=lambda i, x: 50*math.sin(x/10)-100,
             marker='.', colored_by='#ff00ff', label='rose') \
        .add(x=range(100), y=lambda i, x: 50*math.sin(x/10)-200,
             marker='.', colored_by=lambda i, xy: xy[1][i], label='du jaune au rouge') \
        .add(x=range(100), y=lambda i, x: 50*math.sin(x/10)-300,
             marker='x', colored_by=lambda i, xy: ['#ff0000', '#00ff00', '#0000ff'][int(xy[1][i]) % 3],
             label={'#ff0000': 'rouge', '#00ff00': 'vert', '#0000ff': 'bleu'})\
        .add(x=range(100), y=lambda i, x: 50 * math.sin(x / 10) - 400,
             marker='.', markersize=3,
             colored_by=lambda i, xy: '>-400' if xy[1][i] > -400 else '<=-400')
    Plot(3, title=':)').add(x=xgrid(-2, 2, 0.2), y=ygrid(-2, 2, 0.2),
                            z=lambda i, x, y: (x * y) ** 2+8000,
                            marker='.', colored_by=lambda i, xy: xy[2][i], label='du noir au blanc') \
        .add(x=xgrid(-2, 2, 0.2), y=ygrid(-2, 2, 0.2), z=lambda i, x, y: (x * y) ** 2+5000,
             marker='.', colored_by='#ff00ff', label='rose') \
        .add(x=xgrid(-2, 2, 0.2), y=ygrid(-2, 2, 0.2), z=lambda i, x, y: (x * y) ** 2+2000,
             marker='.', colored_by=lambda i, xy: xy[2][i], label='du jaune au rouge') \
        .add(x=xgrid(-2, 2, 0.2), y=ygrid(-2, 2, 0.2), z=lambda i, x, y: (x * y) ** 2-1000,
             marker='x', colored_by=lambda i, xy: ['#ff0000', '#00ff00', '#0000ff'][int(xy[2][i]) % 3],
             label={'#ff0000': 'rouge', '#00ff00': 'vert', '#0000ff': 'bleu'}) \
        .add(x=xgrid(-2, 2, 0.2), y=ygrid(-2, 2, 0.2), z=lambda i, x, y: (x * y) ** 2-4000,
             marker='.', markersize=3,
             colored_by=lambda i, xy: 'exterieur' if math.sqrt(xy[0][i]**2+xy[1][i]**2) > 1 else 'interieur')
    show(True)
