import inspect
import os
import random
import time
from collections import Collection, Iterable
import numpy as np
import re
from IPython.display import display, Markdown, Latex


# #####
# VALUES
# #####

FuncType = type(lambda x: None)
NoneType = type(None)
inf = float('inf')

class Default:
    pass

default = Default()

def is_default(value):
    return value == default



# #####
# INTERN UTILS
# #####

def get_scimple_data_path(path):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scimple_data', path).replace('\\', '/')


# #####
# ERROR
# #####

class ScimpleError(Exception):
    """module specific Errors"""



# #####
# TOOLS
# #####



def is_2d_array_like_not_empty(array):
    """

    :param array: potential 2d array
    :return: bool
    """
    shape = np.shape(array)
    return len(shape) == 2 and shape[0] != 0 and shape[1] != 0


def derivate(f, a, d=10 ** -8, left=False):
    """
    estimate derivate
    :param f: function float -> float
    :param a: point
    :param d: little piece of increment
    :param left: if True, will derivate f from left (from right by default)
    :return:
    """
    return (f(a + d) - f(a)) / d if not left else (f(a) - f(a + d)) / d


def integrate(f, a, b, n_draw=100000, mini=None, maxi=None, n_step=1000):
    """
    Monte Carlo integration
    :param f: function float -> float
    :param a: start value of x for integration
    :param b: end value of x for integration
    :param n_draw: number of random points selections
    :param mini: minimum value of f between a and b (if None, will be calculated)
    :param maxi: maximum value of f between a and b (if None, will be calculated)
    :param n_step: step between points of the curve
    :return:
    """
    res = 0
    if mini is None or maxi is None:
        step = (b - a) / n_step
        courb = [f(x) for x in np.arange(a, b, step)]
        maxi = max(courb)
        mini = min(courb)
    maxi = max(0, maxi)
    mini = min(0, mini)
    for _ in range(n_draw):
        fx = f(random.uniform(a, b))
        tirage = fx
        while tirage == fx:
            tirage = random.uniform(mini, maxi)
        if fx > 0:
            res += 1 if tirage < fx else 0
        else:
            res += 0 if tirage < fx else -1
    return res / n_draw * (b - a) * (maxi - mini)


def flatten_n_times(n, l):
    n = int(n)
    for _ in range(n):
        if any(issubclass(type(elem), Collection) for elem in l):
            res = []
            for elem in l:
                res += list(elem) if issubclass(type(elem), Collection) else [elem]
            l = res
    return l


def try_apply(x, callable_or_collables, default_value=default):
    """

    :param x: any
    :param callable_or_collables: callable or Collection of callables
    :param default: return value if application fail, default None means that x will be returned
    :return:
        callable_or_collables(x) or x if callable_or_collables failed on x
        callable_or_collables[i](x) or x if all callables of callable_or_collables collection failed on x
    """
    if is_default(default_value):
        default_value = x
    if callable(callable_or_collables):
        try:
            return callable_or_collables(x)
        except:
            return default_value
    elif issubclass(type(callable_or_collables), Iterable):
        for fi in callable_or_collables:
            try:
                return fi(x)
            except:
                pass
        return default_value


def nb_params(f):
    """

    :param f: callable
    :return: number of parameters taken by f
    """
    return len(dict(inspect.signature(f).parameters))

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


# ######
# SYSTEM
# ######

def save_environ(path='c:/Prog'):
    """
    Export environment variables in JSON
    :param path: path where the json file wil be created
    :return: full path to created file
    """
    suffixe = time.strftime('%Hh%Mm%Ss_%d_%B_20%y')
    path = os.path.join(path, 'path_' + suffixe + '.json')
    file = open(path, 'w')
    file.write('{\n\t' + str(os.environ)[9:-2].replace('\'', '"') \
               .replace('",', '",\n\t').replace('\\\\', '/') \
               .replace('\t ', '\t') + '\n}')
    file.close()
    print('SUCCESS : Path exported in JSON')
    return path


# ####
# DATA
# ####

def get_sample(id, cast=None):
    dic = {'xyz': "phenyl-Fe-porphyirine-CO2-Me_4_rel.xyz",
           'charges': 'CHARGES_phenyl-Fe-porphyirine-CO2-Me_4_rel',
           'surfaces': 'ek_InTP_CO2_Me_4_graphene_W_r2_k.dat',
           'adults': 'adult.txt'}
    if id == 'xyz':
        res = [line[1:] for line in load_csv(get_scimple_data_path(dic[id]), r'[\t| ]*[[\r\n]|\n]', r'[\t| ]+')[2:-1]]
    elif id == 'charges':
        res = [line[1:] for line in load_csv(get_scimple_data_path(dic[id]), r'[\t| ]*[[\r\n]|\n]', r'[\t| ]+')[2:-1]]
    elif id == 'surfaces':
        res = load_csv(get_scimple_data_path(dic[id]), r'[\t| ]*[[\r\n]|\n]', r'[\t| ]+')
    elif id == 'adults':
        res = load_csv(get_scimple_data_path(dic[id]))
    return cast(res) if cast else res


# #######
# CSV I/O
# #######
def save_csv(path, array_like, delimiter='\n', separator=','):
    """

    :param path: path to file to open (will suppress previous content) or to create
    :param array_like: 2dim array_like (list
    :param delimiter:
    :param separator:
    :return:
    """
    type_value_checks(path, good_types=str, type_message='path should be a string')
    type_value_checks(delimiter, good_types=str, type_message='delimiter should be a string')
    type_value_checks(separator, good_types=str, type_message='separator should be a string')
    type_value_checks(array_like, good_values=lambda array_like: is_2d_array_like_not_empty(array_like),
                      value_message='array-like is not a 2d valid array-like')
    f = open(path, 'w')
    f.write(delimiter.join([separator.join([str(elem) for elem in line]) for line in array_like]))
    f.close()


def load_csv(path, delimiter=r'\n', sep=','):
    type_value_checks(path, good_types=str, type_message='path should be a string')
    type_value_checks(delimiter, good_types=str, type_message='delimiter should be a string')
    type_value_checks(sep, good_types=str, type_message='sep should be a string')
    f = open(path, 'r')
    as_string = f.read()
    f.close()
    return [[try_apply(elem, [int, float]) for elem in re.split(sep, line)] for line in
            re.split(delimiter, as_string)]

# #######
# DISPLAY
# #######

def print_markdown(string):
    """
    Display Markdown in ipynb
    :param string: str
    :return: None
    """
    display(Markdown(string))

def print_latex(string):
    """
    Display Latex in ipynb
    :param string: str
    :return: None
    """
    display(Latex(string))