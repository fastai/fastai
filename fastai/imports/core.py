import csv, gc, gzip, os, pickle, shutil, sys, warnings
import math, matplotlib.pyplot as plt, numpy as np, pandas as pd, random
import scipy.stats, scipy.special
import abc, collections, hashlib, itertools, json, operator
import mimetypes, inspect, typing, functools
import html, re, spacy, requests

from abc import abstractmethod, abstractproperty
from collections import abc,  Counter, defaultdict, Iterable, namedtuple, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from fastprogress import master_bar, progress_bar
from functools import partial, reduce, singledispatch
from IPython.core.debugger import set_trace
from matplotlib import patches, patheffects
from numpy import array, cos, exp, log, sin, tan, tanh
from operator import attrgetter, itemgetter
from pathlib import Path
from spacy.symbols import ORTH
from warnings import warn

#for type annotations
from fastprogress.fastprogress import MasterBar, ProgressBar
from matplotlib.patches import Patch
from numbers import Number
from pandas import Series, DataFrame
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union
from types import SimpleNamespace
