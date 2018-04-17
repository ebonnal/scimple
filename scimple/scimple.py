from __future__ import absolute_import
import numpy as np
import copy
import inspect
import multiprocessing
import os
import re
import sys
import math
import types
import warnings
from random import randint
from threading import Thread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = Axes3D
# -----------------------------------------------------------------------------
# ply: py
#
# Copyright (C) 2001-2017
# David M. Beazley (Dabeaz LLC)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the David Beazley or Dabeaz LLC may be used to
#   endorse or promote products derived from this software without
#  specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

# This tuple contains known string types
try:
    # Python 2.6
    StringTypes = (types.StringType, types.UnicodeType)
except AttributeError:
    # Python 3.0
    StringTypes = (str, bytes)

# This regular expression is used to match valid token names
_is_identifier = re.compile(r'^[a-zA-Z0-9_]+$')


# Exception thrown when invalid token encountered and no default error
# handler is defined.
class LexError(Exception):
    def __init__(self, message, s):
        self.args = (message,)
        self.text = s


# Token class.  This class is used to represent the tokens produced.
class LexToken(object):
    def __str__(self):
        return 'LexToken(%s,%r,%d,%d)' % (self.type, self.value, self.lineno, self.lexpos)

    def __repr__(self):
        return str(self)

    def __bool__(self):
        return False if self.type == 'eof' else True


# This object is a stand-in for a logging object created by the
# logging module.

class PlyLogger(object):
    def __init__(self, f):
        self.f = f

    def critical(self, msg, *args, **kwargs):
        self.f.write((msg % args) + '\n')

    def warning(self, msg, *args, **kwargs):
        self.f.write('WARNING: ' + (msg % args) + '\n')

    def error(self, msg, *args, **kwargs):
        self.f.write('ERROR: ' + (msg % args) + '\n')

    info = critical
    debug = critical


# Null logger is used when no output is generated. Does nothing.
class NullLogger(object):
    def __getattribute__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


# -----------------------------------------------------------------------------
#                        === Lexing Engine ===
#
# The following Lexer class implements the lexer runtime.   There are only
# a few public methods and attributes:
#
#    input()          -  Store a new string in the lexer
#    token()          -  Get the next token
#    clone()          -  Clone the lexer
#
#    lineno           -  Current line number
#    lexpos           -  Current position in the input string
# -----------------------------------------------------------------------------

class Lexer:
    def __init__(self):
        self.lexre = None  # Master regular expression. This is a list of
        # tuples (re, findex) where re is a compiled
        # regular expression and findex is a list
        # mapping regex group numbers to rules
        self.lexretext = None  # Current regular expression strings
        self.lexstatere = {}  # Dictionary mapping lexer states to master regexs
        self.lexstateretext = {}  # Dictionary mapping lexer states to regex strings
        self.lexstaterenames = {}  # Dictionary mapping lexer states to symbol names
        self.lexstate = 'INITIAL'  # Current lexer state
        self.lexstatestack = []  # Stack of lexer states
        self.lexstateinfo = None  # State information
        self.lexstateignore = {}  # Dictionary of ignored characters for each state
        self.lexstateerrorf = {}  # Dictionary of error functions for each state
        self.lexstateeoff = {}  # Dictionary of eof functions for each state
        self.lexreflags = 0  # Optional re compile flags
        self.lexdata = None  # Actual input data (as a string)
        self.lexpos = 0  # Current position in input text
        self.lexlen = 0  # Length of the input text
        self.lexerrorf = None  # Error rule (if any)
        self.lexeoff = None  # EOF rule (if any)
        self.lextokens = None  # List of valid tokens
        self.lexignore = ''  # Ignored characters
        self.lexliterals = ''  # Literal characters that can be passed through
        self.lexmodule = None  # Module
        self.lineno = 1  # Current line number
        self.lexoptimize = False  # Optimized mode

    def clone(self, object=None):
        c = copy.copy(self)

        # If the object parameter has been supplied, it means we are attaching the
        # lexer to a new object.  In this case, we have to rebind all methods in
        # the lexstatere and lexstateerrorf tables.

        if object:
            newtab = {}
            for key, ritem in self.lexstatere.items():
                newre = []
                for cre, findex in ritem:
                    newfindex = []
                    for f in findex:
                        if not f or not f[0]:
                            newfindex.append(f)
                            continue
                        newfindex.append((getattr(object, f[0].__name__), f[1]))
                newre.append((cre, newfindex))
                newtab[key] = newre
            c.lexstatere = newtab
            c.lexstateerrorf = {}
            for key, ef in self.lexstateerrorf.items():
                c.lexstateerrorf[key] = getattr(object, ef.__name__)
            c.lexmodule = object
        return c

    # ------------------------------------------------------------
    # writetab() - Write lexer information to a table file
    # ------------------------------------------------------------
    def writetab(self, lextab, outputdir=''):
        if isinstance(lextab, types.ModuleType):
            raise IOError("Won't overwrite existing lextab module")
        basetabmodule = lextab.split('.')[-1]
        filename = os.path.join(outputdir, basetabmodule) + '.py'
        with open(filename, 'w') as tf:
            tf.write('# %s.py. This file automatically created by PLY (version %s). Don\'t edit!\n' % (
                basetabmodule, __version__))
            tf.write('_tabversion   = %s\n' % repr(__tabversion__))
            tf.write('_lextokens    = set(%s)\n' % repr(tuple(self.lextokens)))
            tf.write('_lexreflags   = %s\n' % repr(self.lexreflags))
            tf.write('_lexliterals  = %s\n' % repr(self.lexliterals))
            tf.write('_lexstateinfo = %s\n' % repr(self.lexstateinfo))

            # Rewrite the lexstatere table, replacing function objects with function names 
            tabre = {}
            for statename, lre in self.lexstatere.items():
                titem = []
                for (pat, func), retext, renames in zip(lre, self.lexstateretext[statename],
                                                        self.lexstaterenames[statename]):
                    titem.append((retext, _funcs_to_names(func, renames)))
                tabre[statename] = titem

            tf.write('_lexstatere   = %s\n' % repr(tabre))
            tf.write('_lexstateignore = %s\n' % repr(self.lexstateignore))

            taberr = {}
            for statename, ef in self.lexstateerrorf.items():
                taberr[statename] = ef.__name__ if ef else None
            tf.write('_lexstateerrorf = %s\n' % repr(taberr))

            tabeof = {}
            for statename, ef in self.lexstateeoff.items():
                tabeof[statename] = ef.__name__ if ef else None
            tf.write('_lexstateeoff = %s\n' % repr(tabeof))

    # ------------------------------------------------------------
    # readtab() - Read lexer information from a tab file
    # ------------------------------------------------------------
    def readtab(self, tabfile, fdict):
        if isinstance(tabfile, types.ModuleType):
            lextab = tabfile
        else:
            exec('import %s' % tabfile)
            lextab = sys.modules[tabfile]

        if getattr(lextab, '_tabversion', '0.0') != __tabversion__:
            raise ImportError('Inconsistent PLY version')

        self.lextokens = lextab._lextokens
        self.lexreflags = lextab._lexreflags
        self.lexliterals = lextab._lexliterals
        self.lextokens_all = self.lextokens | set(self.lexliterals)
        self.lexstateinfo = lextab._lexstateinfo
        self.lexstateignore = lextab._lexstateignore
        self.lexstatere = {}
        self.lexstateretext = {}
        for statename, lre in lextab._lexstatere.items():
            titem = []
            txtitem = []
            for pat, func_name in lre:
                titem.append((re.compile(pat, lextab._lexreflags), _names_to_funcs(func_name, fdict)))

            self.lexstatere[statename] = titem
            self.lexstateretext[statename] = txtitem

        self.lexstateerrorf = {}
        for statename, ef in lextab._lexstateerrorf.items():
            self.lexstateerrorf[statename] = fdict[ef]

        self.lexstateeoff = {}
        for statename, ef in lextab._lexstateeoff.items():
            self.lexstateeoff[statename] = fdict[ef]

        self.begin('INITIAL')

    # ------------------------------------------------------------
    # input() - Push a new string into the lexer
    # ------------------------------------------------------------
    def input(self, s):
        # Pull off the first character to see if s looks like a string
        c = s[:1]
        if not isinstance(c, StringTypes):
            raise ValueError('Expected a string')
        self.lexdata = s
        self.lexpos = 0
        self.lexlen = len(s)

    # ------------------------------------------------------------
    # begin() - Changes the lexing state
    # ------------------------------------------------------------
    def begin(self, state):
        if state not in self.lexstatere:
            raise ValueError('Undefined state')
        self.lexre = self.lexstatere[state]
        self.lexretext = self.lexstateretext[state]
        self.lexignore = self.lexstateignore.get(state, '')
        self.lexerrorf = self.lexstateerrorf.get(state, None)
        self.lexeoff = self.lexstateeoff.get(state, None)
        self.lexstate = state

    # ------------------------------------------------------------
    # push_state() - Changes the lexing state and saves old on stack
    # ------------------------------------------------------------
    def push_state(self, state):
        self.lexstatestack.append(self.lexstate)
        self.begin(state)

    # ------------------------------------------------------------
    # pop_state() - Restores the previous state
    # ------------------------------------------------------------
    def pop_state(self):
        self.begin(self.lexstatestack.pop())

    # ------------------------------------------------------------
    # current_state() - Returns the current lexing state
    # ------------------------------------------------------------
    def current_state(self):
        return self.lexstate

    # ------------------------------------------------------------
    # skip() - Skip ahead n characters
    # ------------------------------------------------------------
    def skip(self, n):
        self.lexpos += n

    # ------------------------------------------------------------
    # opttoken() - Return the next token from the Lexer
    #
    # Note: This function has been carefully implemented to be as fast
    # as possible.  Don't make changes unless you really know what
    # you are doing
    # ------------------------------------------------------------
    def token(self):
        # Make local copies of frequently referenced attributes
        lexpos = self.lexpos
        lexlen = self.lexlen
        lexignore = self.lexignore
        lexdata = self.lexdata

        while lexpos < lexlen:
            # This code provides some short-circuit code for whitespace, tabs, and other ignored characters
            if lexdata[lexpos] in lexignore:
                lexpos += 1
                continue

            # Look for a regular expression match
            for lexre, lexindexfunc in self.lexre:
                m = lexre.match(lexdata, lexpos)
                if not m:
                    continue

                # Create a token for return
                tok = LexToken()
                tok.value = m.group()
                tok.lineno = self.lineno
                tok.lexpos = lexpos

                i = m.lastindex
                func, tok.type = lexindexfunc[i]

                if not func:
                    # If no token type was set, it's an ignored token
                    if tok.type:
                        self.lexpos = m.end()
                        return tok
                    else:
                        lexpos = m.end()
                        break

                lexpos = m.end()

                # If token is processed by a function, call it

                tok.lexer = self  # Set additional attributes useful in token rules
                self.lexmatch = m
                self.lexpos = lexpos

                newtok = func(tok)

                # Every function must return a token, if nothing, we just move to next token
                if not newtok:
                    lexpos = self.lexpos  # This is here in case user has updated lexpos.
                    lexignore = self.lexignore  # This is here in case there was a state change
                    break

                # Verify type of the token.  If not in the token map, raise an error
                if not self.lexoptimize:
                    if newtok.type not in self.lextokens_all:
                        raise LexError("%s:%d: Rule '%s' returned an unknown token type '%s'" % (
                            func.__code__.co_filename, func.__code__.co_firstlineno,
                            func.__name__, newtok.type), lexdata[lexpos:])

                return newtok
            else:
                # No match, see if in literals
                if lexdata[lexpos] in self.lexliterals:
                    tok = LexToken()
                    tok.value = lexdata[lexpos]
                    tok.lineno = self.lineno
                    tok.type = tok.value
                    tok.lexpos = lexpos
                    self.lexpos = lexpos + 1
                    return tok

                # No match. Call t_error() if defined.
                if self.lexerrorf:
                    tok = LexToken()
                    tok.value = self.lexdata[lexpos:]
                    tok.lineno = self.lineno
                    tok.type = 'error'
                    tok.lexer = self
                    tok.lexpos = lexpos
                    self.lexpos = lexpos
                    newtok = self.lexerrorf(tok)
                    if lexpos == self.lexpos:
                        # Error method didn't change text position at all. This is an error.
                        raise LexError("Scanning error. Illegal character '%s'" % (lexdata[lexpos]), lexdata[lexpos:])
                    lexpos = self.lexpos
                    if not newtok:
                        continue
                    return newtok

                self.lexpos = lexpos
                raise LexError("Illegal character '%s' at index %d" % (lexdata[lexpos], lexpos), lexdata[lexpos:])

        if self.lexeoff:
            tok = LexToken()
            tok.type = 'eof'
            tok.value = ''
            tok.lineno = self.lineno
            tok.lexpos = lexpos
            tok.lexer = self
            self.lexpos = lexpos
            newtok = self.lexeoff(tok)
            return newtok

        self.lexpos = lexpos + 1
        if self.lexdata is None:
            raise RuntimeError('No input string given with input()')
        return None

    # Iterator interface
    def __iter__(self):
        return self

    def next(self):
        t = self.token()
        if t is None:
            raise StopIteration
        return t

    __next__ = next


# -----------------------------------------------------------------------------
#                           ==== Lex Builder ===
#
# The functions and classes below are used to collect lexing information
# and build a Lexer object from it.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# _get_regex(func)
#
# Returns the regular expression assigned to a function either as a doc string
# or as a .regex attribute attached by the @TOKEN decorator.
# -----------------------------------------------------------------------------
def _get_regex(func):
    return getattr(func, 'regex', func.__doc__)


# -----------------------------------------------------------------------------
# get_caller_module_dict()
#
# This function returns a dictionary containing all of the symbols defined within
# a caller further down the call stack.  This is used to get the environment
# associated with the yacc() call if none was provided.
# -----------------------------------------------------------------------------
def get_caller_module_dict(levels):
    f = sys._getframe(levels)
    ldict = f.f_globals.copy()
    if f.f_globals != f.f_locals:
        ldict.update(f.f_locals)
    return ldict


# -----------------------------------------------------------------------------
# _funcs_to_names()
#
# Given a list of regular expression functions, this converts it to a list
# suitable for output to a table file
# -----------------------------------------------------------------------------
def _funcs_to_names(funclist, namelist):
    result = []
    for f, name in zip(funclist, namelist):
        if f and f[0]:
            result.append((name, f[1]))
        else:
            result.append(f)
    return result


# -----------------------------------------------------------------------------
# _names_to_funcs()
#
# Given a list of regular expression function names, this converts it back to
# functions.
# -----------------------------------------------------------------------------
def _names_to_funcs(namelist, fdict):
    result = []
    for n in namelist:
        if n and n[0]:
            result.append((fdict[n[0]], n[1]))
        else:
            result.append(n)
    return result


# -----------------------------------------------------------------------------
# _form_master_re()
#
# This function takes a list of all of the regex components and attempts to
# form the master regular expression.  Given limitations in the Python re
# module, it may be necessary to break the master regex into separate expressions.
# -----------------------------------------------------------------------------
def _form_master_re(relist, reflags, ldict, toknames):
    if not relist:
        return []
    regex = '|'.join(relist)
    try:
        lexre = re.compile(regex, reflags)

        # Build the index to function map for the matching engine
        lexindexfunc = [None] * (max(lexre.groupindex.values()) + 1)
        lexindexnames = lexindexfunc[:]

        for f, i in lexre.groupindex.items():
            handle = ldict.get(f, None)
            if type(handle) in (types.FunctionType, types.MethodType):
                lexindexfunc[i] = (handle, toknames[f])
                lexindexnames[i] = f
            elif handle is not None:
                lexindexnames[i] = f
                if f.find('ignore_') > 0:
                    lexindexfunc[i] = (None, None)
                else:
                    lexindexfunc[i] = (None, toknames[f])

        return [(lexre, lexindexfunc)], [regex], [lexindexnames]
    except Exception:
        m = int(len(relist) / 2)
        if m == 0:
            m = 1
        llist, lre, lnames = _form_master_re(relist[:m], reflags, ldict, toknames)
        rlist, rre, rnames = _form_master_re(relist[m:], reflags, ldict, toknames)
        return (llist + rlist), (lre + rre), (lnames + rnames)


# -----------------------------------------------------------------------------
# def _statetoken(s,names)
#
# Given a declaration name s of the form "t_" and a dictionary whose keys are
# state names, this function returns a tuple (states,tokenname) where states
# is a tuple of state names and tokenname is the name of the token.  For example,
# calling this with s = "t_foo_bar_SPAM" might return (('foo','bar'),'SPAM')
# -----------------------------------------------------------------------------
def _statetoken(s, names):
    nonstate = 1
    parts = s.split('_')
    for i, part in enumerate(parts[1:], 1):
        if part not in names and part != 'ANY':
            break

    if i > 1:
        states = tuple(parts[1:i])
    else:
        states = ('INITIAL',)

    if 'ANY' in states:
        states = tuple(names)

    tokenname = '_'.join(parts[i:])
    return (states, tokenname)


# -----------------------------------------------------------------------------
# LexerReflect()
#
# This class represents information needed to build a lexer as extracted from a
# user's input file.
# -----------------------------------------------------------------------------
class LexerReflect(object):
    def __init__(self, ldict, log=None, reflags=0):
        self.ldict = ldict
        self.error_func = None
        self.tokens = []
        self.reflags = reflags
        self.stateinfo = {'INITIAL': 'inclusive'}
        self.modules = set()
        self.error = False
        self.log = PlyLogger(sys.stderr) if log is None else log

    # Get all of the basic information
    def get_all(self):
        self.get_tokens()
        self.get_literals()
        self.get_states()
        self.get_rules()

    # Validate all of the information
    def validate_all(self):
        self.validate_tokens()
        self.validate_literals()
        self.validate_rules()
        return self.error

    # Get the tokens map
    def get_tokens(self):
        tokens = self.ldict.get('tokens', None)
        if not tokens:
            self.log.error('No token list is defined')
            self.error = True
            return

        if not isinstance(tokens, (list, tuple)):
            self.log.error('tokens must be a list or tuple')
            self.error = True
            return

        if not tokens:
            self.log.error('tokens is empty')
            self.error = True
            return

        self.tokens = tokens

    # Validate the tokens
    def validate_tokens(self):
        terminals = {}
        for n in self.tokens:
            if not _is_identifier.match(n):
                self.log.error("Bad token name '%s'", n)
                self.error = True
            if n in terminals:
                self.log.warning("Token '%s' multiply defined", n)
            terminals[n] = 1

    # Get the literals specifier
    def get_literals(self):
        self.literals = self.ldict.get('literals', '')
        if not self.literals:
            self.literals = ''

    # Validate literals
    def validate_literals(self):
        try:
            for c in self.literals:
                if not isinstance(c, StringTypes) or len(c) > 1:
                    self.log.error('Invalid literal %s. Must be a single character', repr(c))
                    self.error = True

        except TypeError:
            self.log.error('Invalid literals specification. literals must be a sequence of characters')
            self.error = True

    def get_states(self):
        self.states = self.ldict.get('states', None)
        # Build statemap
        if self.states:
            if not isinstance(self.states, (tuple, list)):
                self.log.error('states must be defined as a tuple or list')
                self.error = True
            else:
                for s in self.states:
                    if not isinstance(s, tuple) or len(s) != 2:
                        self.log.error("Invalid state specifier %s. Must be a tuple (statename,'exclusive|inclusive')",
                                       repr(s))
                        self.error = True
                        continue
                    name, statetype = s
                    if not isinstance(name, StringTypes):
                        self.log.error('State name %s must be a string', repr(name))
                        self.error = True
                        continue
                    if not (statetype == 'inclusive' or statetype == 'exclusive'):
                        self.log.error("State type for state %s must be 'inclusive' or 'exclusive'", name)
                        self.error = True
                        continue
                    if name in self.stateinfo:
                        self.log.error("State '%s' already defined", name)
                        self.error = True
                        continue
                    self.stateinfo[name] = statetype

    # Get all of the symbols with a t_ prefix and sort them into various
    # categories (functions, strings, error functions, and ignore characters)

    def get_rules(self):
        tsymbols = [f for f in self.ldict if f[:2] == 't_']

        # Now build up a list of functions and a list of strings
        self.toknames = {}  # Mapping of symbols to token names
        self.funcsym = {}  # Symbols defined as functions
        self.strsym = {}  # Symbols defined as strings
        self.ignore = {}  # Ignore strings by state
        self.errorf = {}  # Error functions by state
        self.eoff = {}  # EOF functions by state

        for s in self.stateinfo:
            self.funcsym[s] = []
            self.strsym[s] = []

        if len(tsymbols) == 0:
            self.log.error('No rules of the form t_rulename are defined')
            self.error = True
            return

        for f in tsymbols:
            t = self.ldict[f]
            states, tokname = _statetoken(f, self.stateinfo)
            self.toknames[f] = tokname

            if hasattr(t, '__call__'):
                if tokname == 'error':
                    for s in states:
                        self.errorf[s] = t
                elif tokname == 'eof':
                    for s in states:
                        self.eoff[s] = t
                elif tokname == 'ignore':
                    line = t.__code__.co_firstlineno
                    file = t.__code__.co_filename
                    self.log.error("%s:%d: Rule '%s' must be defined as a string", file, line, t.__name__)
                    self.error = True
                else:
                    for s in states:
                        self.funcsym[s].append((f, t))
            elif isinstance(t, StringTypes):
                if tokname == 'ignore':
                    for s in states:
                        self.ignore[s] = t
                    if '\\' in t:
                        self.log.warning("%s contains a literal backslash '\\'", f)

                elif tokname == 'error':
                    self.log.error("Rule '%s' must be defined as a function", f)
                    self.error = True
                else:
                    for s in states:
                        self.strsym[s].append((f, t))
            else:
                self.log.error('%s not defined as a function or string', f)
                self.error = True

        # Sort the functions by line number
        for f in self.funcsym.values():
            f.sort(key=lambda x: x[1].__code__.co_firstlineno)

        # Sort the strings by regular expression length
        for s in self.strsym.values():
            s.sort(key=lambda x: len(x[1]), reverse=True)

    # Validate all of the t_rules collected
    def validate_rules(self):
        for state in self.stateinfo:
            # Validate all rules defined by functions

            for fname, f in self.funcsym[state]:
                line = f.__code__.co_firstlineno
                file = f.__code__.co_filename
                module = inspect.getmodule(f)
                self.modules.add(module)

                tokname = self.toknames[fname]
                if isinstance(f, types.MethodType):
                    reqargs = 2
                else:
                    reqargs = 1
                nargs = f.__code__.co_argcount
                if nargs > reqargs:
                    self.log.error("%s:%d: Rule '%s' has too many arguments", file, line, f.__name__)
                    self.error = True
                    continue

                if nargs < reqargs:
                    self.log.error("%s:%d: Rule '%s' requires an argument", file, line, f.__name__)
                    self.error = True
                    continue

                if not _get_regex(f):
                    self.log.error("%s:%d: No regular expression defined for rule '%s'", file, line, f.__name__)
                    self.error = True
                    continue

                try:
                    c = re.compile('(?P<%s>%s)' % (fname, _get_regex(f)), self.reflags)
                    if c.match(''):
                        self.log.error("%s:%d: Regular expression for rule '%s' matches empty string", file, line,
                                       f.__name__)
                        self.error = True
                except re.error as e:
                    self.log.error("%s:%d: Invalid regular expression for rule '%s'. %s", file, line, f.__name__, e)
                    if '#' in _get_regex(f):
                        self.log.error("%s:%d. Make sure '#' in rule '%s' is escaped with '\\#'", file, line,
                                       f.__name__)
                    self.error = True

            # Validate all rules defined by strings
            for name, r in self.strsym[state]:
                tokname = self.toknames[name]
                if tokname == 'error':
                    self.log.error("Rule '%s' must be defined as a function", name)
                    self.error = True
                    continue

                if tokname not in self.tokens and tokname.find('ignore_') < 0:
                    self.log.error("Rule '%s' defined for an unspecified token %s", name, tokname)
                    self.error = True
                    continue

                try:
                    c = re.compile('(?P<%s>%s)' % (name, r), self.reflags)
                    if (c.match('')):
                        self.log.error("Regular expression for rule '%s' matches empty string", name)
                        self.error = True
                except re.error as e:
                    self.log.error("Invalid regular expression for rule '%s'. %s", name, e)
                    if '#' in r:
                        self.log.error("Make sure '#' in rule '%s' is escaped with '\\#'", name)
                    self.error = True

            if not self.funcsym[state] and not self.strsym[state]:
                self.log.error("No rules defined for state '%s'", state)
                self.error = True

            # Validate the error function
            efunc = self.errorf.get(state, None)
            if efunc:
                f = efunc
                line = f.__code__.co_firstlineno
                file = f.__code__.co_filename
                module = inspect.getmodule(f)
                self.modules.add(module)

                if isinstance(f, types.MethodType):
                    reqargs = 2
                else:
                    reqargs = 1
                nargs = f.__code__.co_argcount
                if nargs > reqargs:
                    self.log.error("%s:%d: Rule '%s' has too many arguments", file, line, f.__name__)
                    self.error = True

                if nargs < reqargs:
                    self.log.error("%s:%d: Rule '%s' requires an argument", file, line, f.__name__)
                    self.error = True

        for module in self.modules:
            self.validate_module(module)

    # -----------------------------------------------------------------------------
    # validate_module()
    #
    # This checks to see if there are duplicated t_rulename() functions or strings
    # in the parser input file.  This is done using a simple regular expression
    # match on each line in the source code of the given module.
    # -----------------------------------------------------------------------------

    def validate_module(self, module):
        try:
            lines, linen = inspect.getsourcelines(module)
        except IOError:
            return

        fre = re.compile(r'\s*def\s+(t_[a-zA-Z_0-9]*)\(')
        sre = re.compile(r'\s*(t_[a-zA-Z_0-9]*)\s*=')

        counthash = {}
        linen += 1
        for line in lines:
            m = fre.match(line)
            if not m:
                m = sre.match(line)
            if m:
                name = m.group(1)
                prev = counthash.get(name)
                if not prev:
                    counthash[name] = linen
                else:
                    filename = inspect.getsourcefile(module)
                    self.log.error('%s:%d: Rule %s redefined. Previously defined on line %d', filename, linen, name,
                                   prev)
                    self.error = True
            linen += 1


# -----------------------------------------------------------------------------
# lex(module)
#
# Build all of the regular expression rules from definitions in the supplied module
# -----------------------------------------------------------------------------
def lex(module=None, object=None, debug=False, optimize=False, lextab='lextab',
        reflags=int(re.VERBOSE), nowarn=False, outputdir=None, debuglog=None, errorlog=None):
    if lextab is None:
        lextab = 'lextab'

    global lexer

    ldict = None
    stateinfo = {'INITIAL': 'inclusive'}
    lexobj = Lexer()
    lexobj.lexoptimize = optimize
    global token, input

    if errorlog is None:
        errorlog = PlyLogger(sys.stderr)

    if debug:
        if debuglog is None:
            debuglog = PlyLogger(sys.stderr)

    # Get the module dictionary used for the lexer
    if object:
        module = object

    # Get the module dictionary used for the parser
    if module:
        _items = [(k, getattr(module, k)) for k in dir(module)]
        ldict = dict(_items)
        # If no __file__ attribute is available, try to obtain it from the __module__ instead
        if '__file__' not in ldict:
            ldict['__file__'] = sys.modules[ldict['__module__']].__file__
    else:
        ldict = get_caller_module_dict(2)

    # Determine if the module is package of a package or not.
    # If so, fix the tabmodule setting so that tables load correctly
    pkg = ldict.get('__package__')
    if pkg and isinstance(lextab, str):
        if '.' not in lextab:
            lextab = pkg + '.' + lextab

    # Collect parser information from the dictionary
    linfo = LexerReflect(ldict, log=errorlog, reflags=reflags)
    linfo.get_all()
    if not optimize:
        if linfo.validate_all():
            raise SyntaxError("Can't build lexer")

    if optimize and lextab:
        try:
            lexobj.readtab(lextab, ldict)
            token = lexobj.token
            input = lexobj.input
            lexer = lexobj
            return lexobj

        except ImportError:
            pass

    # Dump some basic debugging information
    if debug:
        debuglog.info('lex: tokens   = %r', linfo.tokens)
        debuglog.info('lex: literals = %r', linfo.literals)
        debuglog.info('lex: states   = %r', linfo.stateinfo)

    # Build a dictionary of valid token names
    lexobj.lextokens = set()
    for n in linfo.tokens:
        lexobj.lextokens.add(n)

    # Get literals specification
    if isinstance(linfo.literals, (list, tuple)):
        lexobj.lexliterals = type(linfo.literals[0])().join(linfo.literals)
    else:
        lexobj.lexliterals = linfo.literals

    lexobj.lextokens_all = lexobj.lextokens | set(lexobj.lexliterals)

    # Get the stateinfo dictionary
    stateinfo = linfo.stateinfo

    regexs = {}
    # Build the master regular expressions
    for state in stateinfo:
        regex_list = []

        # Add rules defined by functions first
        for fname, f in linfo.funcsym[state]:
            line = f.__code__.co_firstlineno
            file = f.__code__.co_filename
            regex_list.append('(?P<%s>%s)' % (fname, _get_regex(f)))
            if debug:
                debuglog.info("lex: Adding rule %s -> '%s' (state '%s')", fname, _get_regex(f), state)

        # Now add all of the simple rules
        for name, r in linfo.strsym[state]:
            regex_list.append('(?P<%s>%s)' % (name, r))
            if debug:
                debuglog.info("lex: Adding rule %s -> '%s' (state '%s')", name, r, state)

        regexs[state] = regex_list

    # Build the master regular expressions

    if debug:
        debuglog.info('lex: ==== MASTER REGEXS FOLLOW ====')

    for state in regexs:
        lexre, re_text, re_names = _form_master_re(regexs[state], reflags, ldict, linfo.toknames)
        lexobj.lexstatere[state] = lexre
        lexobj.lexstateretext[state] = re_text
        lexobj.lexstaterenames[state] = re_names
        if debug:
            for i, text in enumerate(re_text):
                debuglog.info("lex: state '%s' : regex[%d] = '%s'", state, i, text)

    # For inclusive states, we need to add the regular expressions from the INITIAL state
    for state, stype in stateinfo.items():
        if state != 'INITIAL' and stype == 'inclusive':
            lexobj.lexstatere[state].extend(lexobj.lexstatere['INITIAL'])
            lexobj.lexstateretext[state].extend(lexobj.lexstateretext['INITIAL'])
            lexobj.lexstaterenames[state].extend(lexobj.lexstaterenames['INITIAL'])

    lexobj.lexstateinfo = stateinfo
    lexobj.lexre = lexobj.lexstatere['INITIAL']
    lexobj.lexretext = lexobj.lexstateretext['INITIAL']
    lexobj.lexreflags = reflags

    # Set up ignore variables
    lexobj.lexstateignore = linfo.ignore
    lexobj.lexignore = lexobj.lexstateignore.get('INITIAL', '')

    # Set up error functions
    lexobj.lexstateerrorf = linfo.errorf
    lexobj.lexerrorf = linfo.errorf.get('INITIAL', None)
    if not lexobj.lexerrorf:
        errorlog.warning('No t_error rule is defined')

    # Set up eof functions
    lexobj.lexstateeoff = linfo.eoff
    lexobj.lexeoff = linfo.eoff.get('INITIAL', None)

    # Check state information for ignore and error rules
    for s, stype in stateinfo.items():
        if stype == 'exclusive':
            if s not in linfo.errorf:
                errorlog.warning("No error rule is defined for exclusive state '%s'", s)
            if s not in linfo.ignore and lexobj.lexignore:
                errorlog.warning("No ignore rule is defined for exclusive state '%s'", s)
        elif stype == 'inclusive':
            if s not in linfo.errorf:
                linfo.errorf[s] = linfo.errorf.get('INITIAL', None)
            if s not in linfo.ignore:
                linfo.ignore[s] = linfo.ignore.get('INITIAL', '')

    # Create global versions of the token() and input() functions
    token = lexobj.token
    input = lexobj.input
    lexer = lexobj

    # If in optimize mode, we write the lextab
    if lextab and optimize:
        if outputdir is None:
            # If no output directory is set, the location of the output files
            # is determined according to the following rules:
            #     - If lextab specifies a package, files go into that package directory
            #     - Otherwise, files go in the same directory as the specifying module
            if isinstance(lextab, types.ModuleType):
                srcfile = lextab.__file__
            else:
                if '.' not in lextab:
                    srcfile = ldict['__file__']
                else:
                    parts = lextab.split('.')
                    pkgname = '.'.join(parts[:-1])
                    exec('import %s' % pkgname)
                    srcfile = getattr(sys.modules[pkgname], '__file__', '')
            outputdir = os.path.dirname(srcfile)
        try:
            lexobj.writetab(lextab, outputdir)
        except IOError as e:
            errorlog.warning("Couldn't write lextab module %r. %s" % (lextab, e))

    return lexobj


# -----------------------------------------------------------------------------
# runmain()
#
# This runs the lexer as a main program
# -----------------------------------------------------------------------------

def runmain(lexer=None, data=None):
    if not data:
        try:
            filename = sys.argv[1]
            f = open(filename)
            data = f.read()
            f.close()
        except IndexError:
            sys.stdout.write('Reading from standard input (type EOF to end):\n')
            data = sys.stdin.read()

    if lexer:
        _input = lexer.input
    else:
        _input = input
    _input(data)
    if lexer:
        _token = lexer.token
    else:
        _token = token

    while True:
        tok = _token()
        if not tok:
            break
        sys.stdout.write('(%s,%r,%d,%d)\n' % (tok.type, tok.value, tok.lineno, tok.lexpos))


# -----------------------------------------------------------------------------
# @TOKEN(regex)
#
# This decorator function can be used to set the regex expression on a function
# when its docstring might need to be set in an alternative way
# -----------------------------------------------------------------------------

def TOKEN(r):
    def set_regex(f):
        if hasattr(r, '__call__'):
            f.regex = _get_regex(r)
        else:
            f.regex = r
        return f

    return set_regex


# Alternative spelling of the TOKEN decorator
Token = TOKEN

warnings.filterwarnings("ignore")


def randomColor(racinecubiquesup, pas):
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


class Plot:
    def __init__(self, dim=2, title="", borders=None, xlabel="", ylabel="", zlabel=""):
        self._atLeastOneLabelDefined = False
        plt.rcParams['lines.color'] = 'b'
        if dim == 2:
            self._fig, self._ax = plt.subplots()
            self._ax.set_title(title)
            if type(borders) == list:
                if len(borders) == 4:
                    self._ax.set_xlim(borders[0], borders[1])
                    self._ax.set_ylim(borders[2], borders[3])
                else:
                    print("length of borders list for 2D plot must be 4")
                    raise Exception()
        elif dim == 3:
            self._fig = plt.figure()
            self._ax = self._fig.gca(projection='3d')
            self._ax.set_title(title)
            self._ax.set_xlabel(xlabel)
            self._ax.set_ylabel(ylabel)
            self._ax.set_zlabel(zlabel)
            if type(borders) == list:
                if len(borders) == 6:
                    self._ax.set_xlim(borders[0], borders[1])
                    self._ax.set_ylim(borders[2], borders[3])
                    self._ax.set_zlim(borders[4], borders[5])
                else:
                    print("length of borders list for 3D plot must be 6")
                    raise Exception()
        else:
            print("SCIMPLE ERROR : in Plot(dim), dim must be 2 or 3")
            raise Exception()
        self._dim = dim  # string
        self._plotables = []

    def add(self, table, xColNum, yColNum, zColNum=None, label="" \
            , color=None, coloredBy=None, plotType='o', markersize=9):
        if self._dim == 2:
            if zColNum != None:
                print("SCIMPLE ERROR : z column declaration for 2D plot forbidden")
                raise Exception()
            if label != "":
                self._atLeastOneLabelDefined = True
            X, Y = [], []
            for lineIndex in range(0, len(table)):
                if len(table[lineIndex]) > max(xColNum, yColNum):
                    X.append(table[lineIndex][xColNum])
                    Y.append(table[lineIndex][yColNum])
            if coloredBy != None:
                plt.plot(X, Y, plotType, label=label, color=coloredBy, markersize=markersize)
            else:
                plt.plot(X, Y, plotType, label=label, markersize=markersize)

            if self._atLeastOneLabelDefined:
                plt.legend(loc='upper right', shadow=True).draggable()

        else:
            if zColNum is None:
                print("SCIMPLE ERROR : z column declaration required for 3D plot")
                raise Exception()
            if type(coloredBy) == int:  # INT COLNUM
                self._atLeastOneLabelDefined = True
                # build groupsDico
                groupsDic = {}
                for line in table:
                    if line[coloredBy] in groupsDic:
                        groupsDic[line[coloredBy]] += [line]
                    else:
                        groupsDic[line[coloredBy]] = [line]
                racinecubiquesup = 0
                while racinecubiquesup ** 3 - racinecubiquesup <= len(groupsDic):
                    racinecubiquesup += 1
                pas = 255 // (racinecubiquesup - 1)
                listOfUsedColors = []
                for group in groupsDic:
                    table = groupsDic[group]
                    X, Y, Z = [], [], []
                    for lineIndex in range(0, len(table)):
                        if len(table[lineIndex]) > max(xColNum, yColNum, zColNum):
                            X.append(table[lineIndex][xColNum])
                            Y.append(table[lineIndex][yColNum])
                            Z.append(table[lineIndex][zColNum])
                    groupColor = randomColor(racinecubiquesup, pas)
                    while groupColor in listOfUsedColors:
                        groupColor = randomColor(racinecubiquesup, pas)
                    listOfUsedColors.append(groupColor)
                    self._ax.plot(X, Y, Z, plotType, label=str(group), color=groupColor, markersize=markersize)
            elif str(type(
                    coloredBy)) == "<class 'function'>":  # and type(coloredBy(1,table[0]))==int :#lineNum,lineList -> int indicateur

                maxi = None
                mini = None
                for i in range(len(table)):
                    try:
                        value = coloredBy(i, table[i])
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

                if label != "":
                    self._atLeastOneLabelDefined = True
                colorDico = {}  # hexa -> plotable lines
                for lineIndex in range(0, len(table)):
                    if len(table[lineIndex]) > max(xColNum, yColNum, zColNum):
                        colorRes = coloredBy(lineIndex, table[lineIndex])
                        deux = colorRes - mini
                        maxolo = max(0, deux)
                        minolo = min(255, maxolo * 255 / (maxi - mini))
                        colorHexaUnit = hex(int(minolo))[2:]
                        if len(colorHexaUnit) == 1:
                            colorHexaUnit = "0" + colorHexaUnit
                        color = "#" + colorHexaUnit * 3
                        if color in colorDico:
                            colorDico[color][0] += [table[lineIndex][xColNum]]
                            colorDico[color][1] += [table[lineIndex][yColNum]]
                            colorDico[color][2] += [table[lineIndex][zColNum]]
                        else:
                            colorDico[color] = [[table[lineIndex][xColNum]], \
                                                [table[lineIndex][yColNum]], [table[lineIndex][zColNum]]]

                legendOn = True
                for colorGroup in colorDico:
                    self._ax.plot(colorDico[colorGroup][0], colorDico[colorGroup][1], \
                                  colorDico[colorGroup][2], plotType, label=(label if legendOn else ""),
                                  color=colorGroup, markersize=markersize, solid_capstyle="round")
                    legendOn = False


            elif type(coloredBy) == str or coloredBy == None:  # simple color field provided or nothing
                if label != "":
                    self._atLeastOneLabelDefined = True
                X, Y, Z = [], [], []
                for lineIndex in range(0, len(table)):
                    if len(table[lineIndex]) > max(xColNum, yColNum, zColNum):
                        X.append(table[lineIndex][xColNum])
                        Y.append(table[lineIndex][yColNum])
                        Z.append(table[lineIndex][zColNum])
                if coloredBy != None:
                    self._ax.plot(X, Y, Z, plotType, label=label, color=coloredBy, markersize=markersize)
                else:
                    self._ax.plot(X, Y, Z, plotType, label=label, markersize=markersize)
            else:

                print("color argument must be function int,List->string ,or string, or int")
                raise Exception()
            if self._atLeastOneLabelDefined:
                self._ax.legend(loc='upper right', shadow=True).draggable()


def show():
    plt.show()


def showAndBlock():
    plt.show(block=True)


class Table:
    def __init__(self, path, firstLine=1, lastLine=None, columnNames=None \
                 , delimiter=r'(\t|[ ])+', newLine=r'(\t| )*((\r\n)|\n)', floatDot='.', numberFormatCharacter='',
                 ignore="", \
                 printTokens=False, printError=False):
        # dev args:
        self._printTokens = printTokens
        self._printError = printError
        # init fields
        self._path = path  # string
        self._firstLine = firstLine  # int
        if (lastLine is not None and lastLine <= 0):
            print("SCIMPLE ERROR : lastLine Argument must be >=1")
            raise Exception()
        self._lastLine = lastLine  # int
        self._columnNames = columnNames  # list
        self._contentAsString = ""  # string
        self._floatTable = []  # string
        self._delimiter = delimiter
        self._newLine = newLine
        self._floatDot = (r'\.' if floatDot == '.' else floatDot)  # regExp
        self._numberFormatCharacter = numberFormatCharacter  # string
        self._ignore = ignore  # regExp
        # MapReduce:
        self._mapping = None
        # import file

        if type(path) is not str:
            try:
                if self._delimiter == r'(\t|[ ])+':
                    self._delimiter = ','
                if self._newLine == r'(\t| )*((\r\n)|\n)':
                    self._newLine = '\n'
                for i in range(len(path)):
                    self._floatTable.append([])
                    for j in range(len(path[i])):
                        self._floatTable[i].append(path[i][j])
                        self._contentAsString += str(path[i][j])
                        if j != len(path[i]) - 1:
                            self._contentAsString += self._delimiter
                    if i != len(path) - 1:
                        self._contentAsString += self._newLine
                print('input considered as array-like')


            except Exception:
                print('Unsupported array-like object in input')
                raise
        else:
            try:
                inFile = open(path, 'r')
                self._contentAsString = inFile.read()
                inFile.close()
                self._parse()
                print('input considered as path to file')
            except IOError:
                self._contentAsString = path
                self._parse()
                print('input considered as string content')


    def __str__(self):
        return self.getString()

    def __unicode__(self):
        return self.getString()

    def __repr__(self):
        return str(self.getTable())

    def __iter__(self):
        return iter(self.getTable())

    def __bool__(self):
        return bool(len(self.getTable()))

    def __getitem__(self, t):
        res = self.getTable()
        if type(t) is int:
            return res[t]
        else:
            for index in t:
                res = res[index]
            return res

    def __delitem__(self, t):
        res = self.getTable()
        if type(t) is int:
            del res[t]
        else:
            for index in t:
                res = res[index]
            del res

    def __len__(self):
        return len(self.getTable())

    def append(self, elem):
        self.getTable().append(list(elem))
        return self

    def pop(self, i=-1):
        popped = self[i]
        del self[i]
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

        t_newLine.__doc__ = self._newLine

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
            if tok.lineno >= self._firstLine and (self._lastLine is None or tok.lineno <= self._lastLine):
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
            elif not (self._lastLine is None or tok.lineno <= self._lastLine):
                break
            if self._printTokens:
                print(tok)
            tok = lexer.token()
        if not tok and (self._lastLine is None or tok.lineno <= self._lastLine):
            currentLine.append(self._try_to_float(currentChars))
            self._floatTable.append(currentLine)

    def _try_to_float(self, s):
        try:
            chars_copied = s
            chars_copied = chars_copied.replace(self._numberFormatCharacter, '')
            if self._floatDot != '\.':
                chars_copied = chars_copied.replace(self._floatDot, '.')
            chars_copied = float(chars_copied)
            if chars_copied % 1 == 0:
                chars_copied = int(chars_copied)
            return chars_copied
        except ValueError:
            return s

    # public :
    def getTable(self):
        """returns the table (list of list) of floats with None for empty fields"""
        return self._floatTable

    def getString(self):
        self._contentAsString = ""
        if self._delimiter == r'(\t|[ ])+':
            self._delimiter = ','
        if self._newLine == r'(\t| )*((\r\n)|\n)':
            self._newLine = '\n'
        self._newLine = self._newLine.replace("\\n","\n").replace("\\t","\t")
        self._delimiter = self._delimiter.replace("\\n","\n").replace("\\t","\t")

        self._contentAsString = self._newLine.join([self._delimiter.join([str(elem) for elem in line]) for line in self])

        return self._contentAsString

    def getCopy(self):
        return copy.deepcopy(self)

    # export
    def save(self, path):
        f = open(path, 'w')
        f.write(self.getString())

    # MapReduce :
    def _init_mapping(self):
        if self._mapping == None:
            self._mapping = dict()
            for lineNum in range(len(self)):
                self._mapping[lineNum + 1] = [self[lineNum]]

    def getMapping(self):
        self._init_mapping()
        return self._mapping

    def getMappingAsTable(self, flatten=False):
        return [[key, self.getMapping()[key]] if not flatten else [key] + self.getMapping()[key]
                for key in self.getMapping()]

    def _build_mr_task(self, key, value_s, f, new_mapping, multi):
        if multi:
            newpairs = f(key, value_s)
            for newkey, newvalue in newpairs:
                if newkey in new_mapping:
                    new_mapping[newkey].append(newvalue)
                else:
                    new_mapping[newkey] = [newvalue]
        else:
            newkey, newvalue = f(key, value_s)
            if newkey in new_mapping:
                new_mapping[newkey].append(newvalue)
            else:
                new_mapping[newkey] = [newvalue]
    class _MRThread(Thread):
        def __init__(self,type,keys,f,new_mapping,multi,table):
            Thread.__init__(self)
            self._type=type
            self._keys=keys
            self._f=f
            self._new_mapping=new_mapping
            self._multi=multi
            self._table=table
        def run(self):
            if self._type == 'map':
                for key in self._keys:
                    for value in self._table._mapping[key]:
                        self._table._build_mr_task(key, value, self._f, self._new_mapping, self._multi)
            elif self._type == 'reduce':
                for key in self._keys:
                    self._table._build_mr_task(key, self._table._mapping[key], self._f, self._new_mapping, self._multi)
            else:
                print("error code 62786289629")
    def _melt_mappings(self,mappings):
        print("here")
        melted_mapping=dict()
        for mapping in mappings:
            for key in mapping:
                if key in melted_mapping:
                    melted_mapping[key] += mapping[key][:]
                else:
                    melted_mapping[key] = mapping[key][:]
        return melted_mapping
    def _perform_map_or_reduce(self,type, f, multi, threads):
        new_mappings = [{} for _ in range(threads)]
        keys = list(self.getMapping().keys()) # _init_mapping() ran during getMapping
        threads = min(threads,len(keys))
        keys_parts = [[] for _ in range(threads)]
        for i in range(len(keys)):
            keys_parts[i%threads].append(keys[i])
        print(555,keys_parts)
        threads_list = list()
        for i in range(threads):
            threads_list.append(self._MRThread(type, keys_parts[i], f, new_mappings[i], multi, self))
        for thread in threads_list:
            thread.start()
        for thread in threads_list:
            thread.join()
        self._mapping = self._melt_mappings(new_mappings)
    def map(self, f, multi=False, threads=multiprocessing.cpu_count()):
        """
        f du type lambda key value : return (key, value)
        :return: self (to chain)
        """
        self._perform_map_or_reduce('map', f, multi, threads)
        return self

    def reduce(self, f, multi=False, threads=multiprocessing.cpu_count()):
        """
        f : lambda key, values_list : key, value
        :param f:
        :return: self (to chain)
        """
        self._perform_map_or_reduce('reduce', f, multi, threads)
        return self


def run_example():
    _ROOT = os.path.abspath(os.path.dirname(__file__))

    # os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scimple_data', path)
    def get_data(path):
        return os.path.join(_ROOT, 'scimple_data', path)

    source = """print("Few Examples Of Scimple Plots :), are they well displayed ? \SOURCE :" + source)
    # example :
    moleculeTable = Table(get_data("phenyl-Fe-porphyirine-CO2-Me_4_rel.xyz"), firstLine=3, lastLine=103)
    grapheneTable = Table(get_data("phenyl-Fe-porphyirine-CO2-Me_4_rel.xyz"), firstLine=104, lastLine=495)
    chargesGraphene = Table(get_data("CHARGES_phenyl-Fe-porphyirine-CO2-Me_4_rel"), firstLine=104, lastLine=495)

    # print(moleculeTable)

    # 3D delta et molec
    def f(lineNum, line):
        # print(sum(chargesGraphene.getTable()[lineNum][1:]) - 4)
        # print(line)
        # print(chargesGraphene.getTable()[lineNum])
        # print(2)
        return sum(chargesGraphene.getTable()[lineNum][1:]) - 4

    myPlot3D = Plot(dim=3, xlabel="X", ylabel="Y", zlabel="Z", borders=[-40, 40, -40, 40, 15, 30],
                    title="Test Graphe #3D delta et molec")
    myPlot3D.add(moleculeTable, xColNum=2, yColNum=3, zColNum=4, markersize=2, coloredBy=1)
    myPlot3D.add(grapheneTable, xColNum=2, yColNum=3, zColNum=4, markersize=2, label="graphene",
                 coloredBy=f)
    # 3D comparatif z et delta:

    myPlot3Dbis = Plot(dim=3, xlabel="X", ylabel="Y", zlabel="Z", borders=[-40, 40, -40, 40, 15, 30],
                       title="Test Graphe #3D comparatif z et delta:")
    myPlot3Dbis.add([grapheneTable.getTable()[i][:4] + [grapheneTable.getTable()[i][4] + 10] for i in
                     range(len(grapheneTable.getTable()) - 1)], xColNum=2, yColNum=3, zColNum=4, label="colored by z",
                    coloredBy=lambda lineNum, line: line[4])
    myPlot3Dbis.add(grapheneTable, xColNum=2, yColNum=3, zColNum=4, label="colored by delta",
                    coloredBy=lambda lineNum, line: (sum(chargesGraphene.getTable()[lineNum][1:]) - 4))

    # 2D:

    myPlot2D = Plot(dim=2, xlabel="X", zlabel="Z", borders=[-20, 20, 18, 19], title="Test Graphe 2D")
    myPlot2D.add(grapheneTable, xColNum=3, yColNum=4, label="graphene Y/Z", coloredBy="#f4a28c", markersize=20)
    myPlot2D.add(grapheneTable, xColNum=2, yColNum=4, label="graphene X/Z", plotType='-')

    # 3D plot 2 surfaces:

    myTable = Table(get_data("ek_InTP_CO2_Me_4_graphene_W_r2_k.dat"), firstLine=1)
    myPlot3Dter = Plot(dim=3, xlabel="X", ylabel="Y", zlabel="Z", title="deux surfaces, point de weyl ?")
    myPlot3Dter.add(myTable, xColNum=0, yColNum=1, zColNum=4, label="column 4", coloredBy="#000000")
    myPlot3Dter.add(myTable, xColNum=0, yColNum=1, zColNum=5, label="column 5")
    showAndBlock()"""
    print("Few Examples Of Scimple Plots :), are they well displayed ? \SOURCE :" + source)
    # example :
    moleculeTable = Table(get_data("phenyl-Fe-porphyirine-CO2-Me_4_rel.xyz"), firstLine=3, lastLine=103)
    grapheneTable = Table(get_data("phenyl-Fe-porphyirine-CO2-Me_4_rel.xyz"), firstLine=104, lastLine=495)
    chargesGraphene = Table(get_data("CHARGES_phenyl-Fe-porphyirine-CO2-Me_4_rel"), firstLine=104, lastLine=495)

    # print(moleculeTable)

    # 3D delta et molec
    def f(lineNum, line):
        # print(sum(chargesGraphene.getTable()[lineNum][1:]) - 4)
        # print(line)
        # print(chargesGraphene.getTable()[lineNum])
        # print(2)
        return sum(chargesGraphene.getTable()[lineNum][1:]) - 4

    myPlot3D = Plot(dim=3, xlabel="X", ylabel="Y", zlabel="Z", borders=[-40, 40, -40, 40, 15, 30],
                    title="Test Graphe #3D delta et molec")
    myPlot3D.add(moleculeTable, xColNum=2, yColNum=3, zColNum=4, markersize=2, coloredBy=1)
    myPlot3D.add(grapheneTable, xColNum=2, yColNum=3, zColNum=4, markersize=2, label="graphene",
                 coloredBy=f)
    """EN TESTS :
    #3D molec avec couleurs standards
    dicoCouleursStandards={'C':"#000000",'H':"#ffffff",'O':'r','N':'b','Fe':"#00ffff"}
    myPlot3D=Plot(dim=3,xlabel="X",ylabel="Y",zlabel="Z",borders=[-40,40,-40,40,15,30],title="Test Graphe #3D molec avec couleurs standards")
    myPlot3D.add(molecTable,xColNum=2,yColNum=3,zColNum=4,coloredBy=lambda lineNum,line:dicoCouleursStandards[line[1]])
    """
    # 3D comparatif z et delta:

    myPlot3Dbis = Plot(dim=3, xlabel="X", ylabel="Y", zlabel="Z", borders=[-40, 40, -40, 40, 15, 30],
                       title="Test Graphe #3D comparatif z et delta:")
    myPlot3Dbis.add([grapheneTable.getTable()[i][:4] + [grapheneTable.getTable()[i][4] + 10] for i in
                     range(len(grapheneTable.getTable()) - 1)], xColNum=2, yColNum=3, zColNum=4, label="colored by z",
                    coloredBy=lambda lineNum, line: line[4])
    myPlot3Dbis.add(grapheneTable, xColNum=2, yColNum=3, zColNum=4, label="colored by delta",
                    coloredBy=lambda lineNum, line: (sum(chargesGraphene.getTable()[lineNum][1:]) - 4))

    # 2D:

    myPlot2D = Plot(dim=2, xlabel="X", zlabel="Z", borders=[-20, 20, 18, 19], title="Test Graphe 2D")
    myPlot2D.add(grapheneTable, xColNum=3, yColNum=4, label="graphene Y/Z", coloredBy="#f4a28c", markersize=20)
    myPlot2D.add(grapheneTable, xColNum=2, yColNum=4, label="graphene X/Z", plotType='-')

    # 3D plot 2 surfaces:

    myTable = Table(get_data("ek_InTP_CO2_Me_4_graphene_W_r2_k.dat"), firstLine=1)
    myPlot3Dter = Plot(dim=3, xlabel="X", ylabel="Y", zlabel="Z", title="deux surfaces, point de weyl ?")
    myPlot3Dter.add(myTable, xColNum=0, yColNum=1, zColNum=4, label="column 4", coloredBy="#000000")
    myPlot3Dter.add(myTable, xColNum=0, yColNum=1, zColNum=5, label="column 5")
    showAndBlock()


if __name__ == '__main__':
    # run_example()
    tab = Table("test.txt", firstLine=3, lastLine=495)

    # print(tab.getMapping())
    # tab.map(lambda lineNum, line: (line[1], line[4])) \
    #     .map(lambda key, value: (key, value + 2)) \
    #     .reduce(lambda key, values_list: (key, sum(values_list) / len(values_list))) \
    #     .map(lambda key, value: (len(key),1)) \
    #     .reduce(lambda key, values_list: (key,sum(values_list)))

    # tab.map(lambda lineNum, line: (line[1],1)) \
    #     .reduce(lambda key, values_list: [(key, sum(values_list))] if isinstance(key,str) else [], multi=True)
    # print(tab.getMapping())
    # print(tab.getMappingAsTable(True))
    # tab = Table(tab.getMappingAsTable(True), delimiter=';')
    # print(tab.getString())
    # tab.append({"fin de file"})
    tab2 = Table(tab,delimiter = ';;').append(["test"]).save('out.txt')
    # tab2 = tab.getCopy()
    # print(tab.pop())
    # print(tab2, tab)
    # print(Table("14 41546 ej;\t zkozf 45 64   ",newLine=r'\t').getString(),Table("14 41546 ej;\n zkozf 45 64   ",newLine='\\n').getTable())
