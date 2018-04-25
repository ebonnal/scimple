from __future__ import absolute_import

import collections
import copy
import multiprocessing
import os
from random import randint
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

_ = Axes3D


class ScimpleError(Exception):
    pass


def type_value_checks(x, type=None, good_values=None, type_message='', value_message=''):
    if type is not None:
        if not isinstance(x, type):
            raise TypeError(type_message)
    if good_values is not None:
        if callable(good_values):
            if not good_values(x):
                raise ValueError(value_message)
        elif isinstance(good_values, collections.Sequence):
            if x not in good_values:
                raise ValueError(value_message)
        else:
            return ValueError("good_values must be either a callable or a sequence")


class Plot:
    _gs = gridspec.GridSpec(2, 2, width_ratios=[50, 1], height_ratios=[1, 50])
    plt.rcParams['lines.color'] = 'b'

    def __init__(self, dim=2, title=None, xlabel=None, ylabel=None, zlabel=None, borders=None, bg_color=None):
        self._at_least_one_label_defined = False
        self._color_bar_nb = 0
        self._fig = plt.figure()
        type_value_checks(dim, type=int, good_values={2, 3},
                          type_message='dim must be an integer',
                          value_message='dim value must be 2 or 3')
        if dim == 2:
            self._ax = self._fig.add_subplot(self._gs[2])
        elif dim == 3:
            self._ax = self._fig.add_subplot(self._gs[2], projection='3d')
        self._dim = dim
        if borders is not None:
            type_value_checks(borders, type={list, tuple}, good_values=lambda borders_: len(borders_) == dim * 2,
                              type_message="borders must be a list or a tuple",
                              value_message="length of borders list for 3D plot must be 6")
            self._ax.set_xlim(borders[0], borders[1])
            self._ax.set_ylim(borders[2], borders[3])
            if dim == 3:
                self._ax.set_zlim(borders[4], borders[5])
        if title is not None:
            type_value_checks(title, type=str, type_message="title type must be str")
            self._ax.set_title(title)
        if xlabel is not None:
            type_value_checks(xlabel, type=str, type_message="xlabel type must be str")
            self._ax.set_xlabel(xlabel)
        if ylabel is not None:
            type_value_checks(ylabel, type=str, type_message="ylabel type must be str")
            self._ax.set_ylabel(ylabel)
        if zlabel is not None:
            type_value_checks(zlabel, type=str, good_values=lambda zlabel_: dim != 3,
                              type_message="zlabel type must be str",
                              value_message="zlabel is only settable on 3D plots")
            self._ax.set_zlabel(zlabel)
        if bg_color is not None:
            type_value_checks(bg_color, type=str, good_values=self.is_hexa_color,
                              type_message="bg_color must be of type str",
                              value_message="bg_color_must be a valid hexadecimal color code, example : '#ff45ab'")
            self._ax.set_facecolor(bg_color)

    @staticmethod
    def is_hexa_color(color):
        if len(color) != 7:
            return False
        if color[0] != '#':
            return False
        try:
            int('0x' + color[1:], 0)
            return True
        except ValueError:
            return False

    @staticmethod
    def _random_color_well_dispatched(racinecubiquesup, pas):
        R = hex(randint(0, racinecubiquesup - 1) * pas)[2:]
        if len(R) == 1:
            R = "0" + R
        G = hex(randint(0, racinecubiquesup - 1) * pas)[2:]
        if len(G) == 1:
            G = "0" + G
        B = hex(randint(0, racinecubiquesup - 1) * pas)[2:]
        if len(B) == 1:
            B = "0" + B
        while R == B and R == G:

            R = hex(randint(0, racinecubiquesup - 1) * pas)[2:]
            if len(R) == 1:
                R = "0" + R
            G = hex(randint(0, racinecubiquesup - 1) * pas)[2:]
            if len(G) == 1:
                G = "0" + G
            B = hex(randint(0, racinecubiquesup - 1) * pas)[2:]
            if len(B) == 1:
                B = "0" + B
        return "#" + R + G + B

    @staticmethod
    def _pastelize(color, coef_pastel=2, coef_fonce=0.75):
        if color[0] != '#':
            color = '#' + color
        colors = [int('0x' + color[1:3], 0), int('0x' + color[3:5], 0), int('0x' + color[5:7], 0)]
        for i in range(len(colors)):
            colors[i] = (colors[i] + (255 - colors[i]) / coef_pastel) * coef_fonce
        return '#' + ''.join(map(lambda x: hex(int(x))[2:], colors))

    def _print_color_bar(self, color_label, mini, maxi):
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
            color_bar_subplot.set_ylabel(color_label)
            color_bar_subplot.yaxis.tick_right()
        else:
            color_bar_subplot.set_yticks([])
            color_bar_subplot.set_xlabel(color_label)
            color_bar_subplot.xaxis.tick_top()

    def add(self, table, x_col_num, y_col_num, z_col_num=None, first_line=0, last_line=None, label="", colored_by=None,
            color_label='', plot_type='o', markersize=9):
        if last_line is None:
            last_line = len(table) - 1
        if self._dim == 2:
            if z_col_num != None:
                print("S_c_i_m_p_l_e E_r_r_o_r : z column declaration for 2D plot forbidden")
                raise Exception()
            if label != "":
                self._at_least_one_label_defined = True
            X, Y = [], []
            for line_index in range(first_line, min(last_line + 1, len(table))):
                if len(table[line_index]) > max(x_col_num, y_col_num):
                    X.append(table[line_index][x_col_num])
                    Y.append(table[line_index][y_col_num])
            if colored_by != None:
                if type(colored_by) is str:
                    plt.plot(X, Y, plot_type, label=label, color=colored_by, markersize=markersize)
                elif type(colored_by) is int:
                    self._at_least_one_label_defined = True
                    # build groups_dico
                    groups_dic = {}
                    for index in range(first_line, min(last_line + 1, len(table))):
                        line = table[index]
                        if line[colored_by] in groups_dic:
                            groups_dic[line[colored_by]] += [line]
                        else:
                            groups_dic[line[colored_by]] = [line]
                    racinecubiquesup = 0
                    while racinecubiquesup ** 3 - racinecubiquesup <= len(groups_dic):
                        racinecubiquesup += 1
                    pas = 255 // (racinecubiquesup - 1)
                    list_of_used_colors = []
                    for group in groups_dic:
                        table = groups_dic[group]
                        X, Y = [], []
                        for line_index in range(first_line, min(last_line + 1, len(table))):
                            if len(table[line_index]) > max(x_col_num, y_col_num):
                                X.append(table[line_index][x_col_num])
                                Y.append(table[line_index][y_col_num])
                        group_color = _random_color_well_dispatched(racinecubiquesup, pas)
                        while group_color in list_of_used_colors:
                            group_color = _random_color_well_dispatched(racinecubiquesup, pas)
                        list_of_used_colors.append(group_color)

                        self._ax.plot(X, Y, plot_type, label=str(group), color=self._pastelize(group_color),
                                      markersize=markersize)
                elif str(type(
                        colored_by)) == "<class 'function'>":  # and type(colored_by(1,table[0]))==int :#line_num,line_list -> int indicateur

                    maxi = None
                    mini = None
                    for i in range(first_line, min(last_line + 1, len(table))):
                        try:
                            value = colored_by(i, table[i])
                        except Exception:
                            value = maxi
                        if maxi is None:
                            maxi = value
                            mini = value
                        try:
                            maxi = max(maxi, value)
                            mini = min(mini, value)
                        except:
                            print(454545)
                    self._print_color_bar(color_label, mini, maxi)

                    if label != "":
                        self._at_least_one_label_defined = True
                    color_dico = {}  # hexa -> plotable lines
                    for line_index in range(first_line, min(last_line + 1, len(table))):
                        if len(table[line_index]) > max(x_col_num, y_col_num):
                            color_res = colored_by(line_index, table[line_index])
                            deux = color_res - mini
                            maxolo = max(0, deux)
                            minolo = min(255, maxolo * 255 / (maxi - mini))
                            color_hexa_unit = hex(int(minolo))[2:]
                            if len(color_hexa_unit) == 1:
                                color_hexa_unit = "0" + color_hexa_unit
                            color = "#" + color_hexa_unit * 3 if self._color_bar_nb == 1 else "#ff" + color_hexa_unit + "00"
                            if color in color_dico:
                                color_dico[color][0] += [table[line_index][x_col_num]]
                                color_dico[color][1] += [table[line_index][y_col_num]]
                            else:
                                color_dico[color] = [[table[line_index][x_col_num]], [table[line_index][y_col_num]]]

                    legend_on = True
                    for color_group in color_dico:
                        self._ax.plot(color_dico[color_group][0], color_dico[color_group][1], plot_type,
                                      label=(label if legend_on else ""),
                                      color=color_group, markersize=markersize, solid_capstyle="round")
                        legend_on = False
            else:
                self._ax.plot(X[first_line:min(last_line + 1, len(table))],
                              Y[first_line:min(last_line + 1, len(table))],
                              plot_type, label=label, markersize=markersize)

            if self._at_least_one_label_defined:
                self._ax.legend(loc='upper right', shadow=True).draggable()

        else:
            if z_col_num is None:
                print("S_c_i_m_p_l_e E_r_r_o_r : z column declaration required for 3D plot")
                raise Exception()
            if type(colored_by) == int:  # I_n_t C_o_l_n_u_m
                self._at_least_one_label_defined = True
                # build groups_dico
                groups_dic = {}
                for index in range(first_line, min(last_line + 1, len(table))):
                    line = table[index]
                    if line[colored_by] in groups_dic:
                        groups_dic[line[colored_by]] += [line]
                    else:
                        groups_dic[line[colored_by]] = [line]
                racinecubiquesup = 0
                while racinecubiquesup ** 3 - racinecubiquesup <= len(groups_dic):
                    racinecubiquesup += 1
                pas = 255 // (racinecubiquesup - 1)
                list_of_used_colors = []
                for group in groups_dic:
                    table = groups_dic[group]
                    X, Y, Z = [], [], []
                    for line_index in range(first_line, min(last_line + 1, len(table))):
                        if len(table[line_index]) > max(x_col_num, y_col_num, z_col_num):
                            X.append(table[line_index][x_col_num])
                            Y.append(table[line_index][y_col_num])
                            Z.append(table[line_index][z_col_num])
                    group_color = _random_color_well_dispatched(racinecubiquesup, pas)
                    while group_color in list_of_used_colors:
                        group_color = _random_color_well_dispatched(racinecubiquesup, pas)
                    list_of_used_colors.append(group_color)
                    self._ax.plot(X, Y, Z, plot_type, label=str(group), color=self._pastelize(group_color),
                                  markersize=markersize)
            elif str(type(
                    colored_by)) == "<class 'function'>":  # and type(colored_by(1,table[0]))==int :#line_num,line_list -> int indicateur
                maxi = None
                mini = None
                for i in range(first_line, min(last_line + 1, len(table))):
                    try:
                        value = colored_by(i, table[i])
                    except Exception:
                        value = maxi
                    if maxi is None:
                        maxi = value
                        mini = value
                    try:
                        maxi = max(maxi, value)
                        mini = min(mini, value)
                    except:
                        print(454545)
                self._print_color_bar(color_label, mini, maxi)

                if label != "":
                    self._at_least_one_label_defined = True
                color_dico = {}  # hexa -> plotable lines
                for line_index in range(first_line, min(last_line + 1, len(table))):
                    if len(table[line_index]) > max(x_col_num, y_col_num, z_col_num):
                        color_res = colored_by(line_index, table[line_index])
                        deux = color_res - mini
                        maxolo = max(0, deux)
                        minolo = min(255, maxolo * 255 / (maxi - mini))
                        color_hexa_unit = hex(int(minolo))[2:]
                        if len(color_hexa_unit) == 1:
                            color_hexa_unit = "0" + color_hexa_unit
                        color = "#" + color_hexa_unit * 3 if self._color_bar_nb == 1 else "#ff" + color_hexa_unit + "00"
                        if color in color_dico:
                            color_dico[color][0] += [table[line_index][x_col_num]]
                            color_dico[color][1] += [table[line_index][y_col_num]]
                            color_dico[color][2] += [table[line_index][z_col_num]]
                        else:
                            color_dico[color] = [[table[line_index][x_col_num]], [table[line_index][y_col_num]],
                                                 [table[line_index][z_col_num]]]

                legend_on = True
                for color_group in color_dico:
                    self._ax.plot(color_dico[color_group][0], color_dico[color_group][1], color_dico[color_group][2],
                                  plot_type, label=(label if legend_on else ""),
                                  color=color_group, markersize=markersize, solid_capstyle="round")
                    legend_on = False


            elif type(colored_by) == str or colored_by == None:  # simple color field provided or nothing
                if label != "":
                    self._at_least_one_label_defined = True
                X, Y, Z = [], [], []
                for line_index in range(first_line, min(last_line + 1, len(table))):
                    if len(table[line_index]) > max(x_col_num, y_col_num, z_col_num):
                        X.append(table[line_index][x_col_num])
                        Y.append(table[line_index][y_col_num])
                        Z.append(table[line_index][z_col_num])
                if colored_by != None:
                    self._ax.plot(X, Y, Z, plot_type, label=label, color=colored_by, markersize=markersize)
                else:
                    self._ax.plot(X, Y, Z, plot_type, label=label, markersize=markersize)
            else:

                print("color argument must be function int,List->string ,or string, or int")
                raise Exception()
            if self._at_least_one_label_defined:
                self._ax.legend(loc='upper right', shadow=True).draggable()
        return self


def show():
    plt.show()


def show_and_block():
    plt.show(block=True)


class Line:
    def __init__(self, line, column_names=None, delimiter=r'(	|[ ])+'):
        self._column_names = column_names
        self._line = list(line)
        self._delimiter = delimiter
        self._content_as_string = ""  # string
        if isinstance(line, Line):
            column_names = list(line.get_columns_names().keys())
        if column_names is None or len(column_names) == 0:
            self._column_names = {}
        else:
            self._column_names = {column_names[i]: i for i in range(len(column_names))}  # list

    def __contains__(self, elem_to_find):
        return elem_to_find in self._list

    def __str__(self):
        return self.get_string()

    def __unicode__(self):
        return self.get_string()

    def __repr__(self):
        return str(self.get_line())

    def __iter__(self):
        return iter(self.get_line())

    def __bool__(self):
        return bool(len(self.get_line()))

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
            return self.get_line()[t]
        if type(t) is str:
            return self.get_line(t)
        if type(t) is tuple:
            t = self._from_col_name_to_int(t)
            return self.get_line(t)

    def __delitem__(self, t):
        if type(t) in {slice, int}:
            del self.get_line()[t]
        if type(t) is str:
            self._filter_by_columns_del_keep('del', t)
        if type(t) is tuple:
            self._filter_by_columns_del_keep('del', t)

    def __len__(self):
        return len(self.get_line())

    def append(self, elem):
        self.get_line().append(list(elem))
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
            popped = self.get_line()[columns]
            del self.get_line()[columns]
        elif type(columns) is str:
            popped = self.get_line(columns)
            self._filter_by_columns_del_keep('del', columns)
        elif type(columns) is tuple:
            popped = self.get_line(columns)
            self._filter_by_columns_del_keep('del', columns)
        return popped

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
        i = 0
        j = 0
        while i < len(self.get_line()):
            if (del_or_keep == 'del' and i in columns) or (del_or_keep == 'keep' and i not in columns):
                del self.get_line()[j]
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

    def get_line(self, columns_num_tuple=None, *columns_num):
        '''returns the table (list of list) of floats with None for empty fields'''
        if columns_num_tuple is not None:
            if type(columns_num_tuple) in {str, int}:
                columns = [columns_num_tuple]
                for name in columns_num:
                    columns.append(name)
            else:
                columns = columns_num_tuple
            columns = self._from_col_name_to_int(columns)
            res = [self._line[i] for i in range(len(self._line)) if i in columns]
            return res
        else:
            return self._line

    def get_string(self, delimiter=None):
        if delimiter == None:
            delimiter = self._delimiter
        self._content_as_string = ""
        if delimiter == r'(	|[ ])+':
            delimiter = ','
        delimiter = delimiter.replace("\n", "").replace("\t", "	")
        self._content_as_string = delimiter.join([str(elem) for elem in self.get_line()])

        return self._content_as_string

    def get_copy(self):
        return copy.deepcopy(self)

    # export
    def save(self, path, delimiter=None):
        f = open(path, 'w')
        f.write(self.get_string(delimiter))


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
        pass

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

    # export
    def save(self, path, delimiter=None, new_line=None):
        f = open(path, 'w')
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


def flatten_n_times(n, l):
    n = int(n)
    for _ in range(n):
        if any(type(elem) is list for elem in l):
            res = []
            for elem in l:
                res += elem if type(elem) is list else [elem]
            l = res
    return l


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
    # os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scimple_data', path)

    source = '''
import scimple as scm
tab = scm.Table(scm.get_sample('xyz'),first_line=2,last_line=494) # 
charges = scm.Table(scm.get_sample('charges'),first_line=1)
scm.Plot(3,bg_color='#ddddff').add(tab[101:],1,2,3,markersize=4,plot_type='o',colored_by =lambda i,_:sum(charges[101+i])).add(tab[:101],1,2,3,markersize=4,plot_type='o',colored_by =0)
scm.Plot(2,bg_color='#cccccc').add(tab[:101],1,2,colored_by=0).add(tab[101:],1,2,markersize=4,plot_type='x',colored_by =lambda i,_:sum(charges[101+i]))
scm.Plot(2,bg_color='#cccccc', xlabel="x axis", ylabel="y axis").add(tab[101:],1,2,markersize=6,plot_type='o',colored_by =lambda _,line:line[3], color_label="z axis").add(tab[101:],1,2,markersize=4,plot_type='x',colored_by =lambda i,_:sum(charges[101+i]), color_label="external electrons")
scm.Plot(2,bg_color='#cccccc', xlabel="atom", ylabel="z axis").add(tab,0,3,markersize=6,plot_type='o',colored_by = 0, color_label="z axis")#scm.show()'''
    print("Few Examples Of Scimple Plots :), are they well displayed ? \S_o_u_r_c_e :" + source)
    # example :

    tab = Table(get_sample('xyz'), first_line=2, last_line=494)  #
    # print(np.array(tab.map(lambda i, line: ('line '+str(i)+':',line)).get_mapping_as_table(2)))
    charges = Table(get_sample('charges'), first_line=1)
    Plot(3, bg_color='#ddddff').add(tab, 1, 2, 3, first_line=101, markersize=4, plot_type='o',
                                    colored_by=lambda i, _: sum(charges[i])).add(tab, 1, 2, 3, last_line=100
                                                                                 , markersize=4, plot_type='o',
                                                                                 colored_by=0)
    Plot(2, bg_color='#cccccc').add(tab, 1, 2, last_line=100, colored_by=0).add(tab, 1, 2, first_line=101, markersize=4,
                                                                                plot_type='x',
                                                                                colored_by=lambda i, _: sum(charges[i]))
    Plot(2, bg_color='#cccccc', xlabel="x axis", ylabel="y axis").add(tab, 1, 2, first_line=101, markersize=6,
                                                                      plot_type='o', colored_by=lambda _, line: line[3],
                                                                      color_label="z axis").add(tab, 1, 2,
                                                                                                first_line=101,
                                                                                                markersize=4,
                                                                                                plot_type='x',
                                                                                                colored_by=lambda i,
                                                                                                                  _: sum(
                                                                                                    charges[i]),
                                                                                                color_label="external electrons")
    Plot(2, bg_color='#cccccc', xlabel="atom", ylabel="z axis").add(tab, 0, 3, markersize=6, plot_type='o',
                                                                    colored_by=0, color_label="z axis")  # show()
    show_and_block()


if __name__ == '__main__':
    # run_example()
    data = get_sample('adults')
    print(data.get_columns_names())
    data['age'] = [500]
    print(data['age'])
    # print(data)
    # print(data.get_columns_names())
    l = Line(Line([1, 2, 4, 45, 64, 'jkgh'], column_names=['a', 'b', 'c', 'd', 'e', 'f'])['b', 'd', 1, 2, 3, 4, 5, 6],
             column_names=['b', 'd'])
    print(l.pop('d'))
    print(l.get_line())
