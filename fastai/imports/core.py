import csv, gc, gzip, os, pickle, shutil, sys, warnings, yaml
import math, matplotlib.pyplot as plt, numpy as np, pandas as pd, random
import scipy.stats, scipy.special
import abc, collections, hashlib, itertools, json, operator, pathlib
import mimetypes, inspect, typing, functools, importlib
import html, re, spacy, requests, tarfile, numbers

from abc import abstractmethod, abstractproperty
from collections import abc,  Counter, defaultdict, Iterable, namedtuple, OrderedDict
import concurrent
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from functools import partial, reduce
from pdb import set_trace
from matplotlib import patches, patheffects
from numpy import array, cos, exp, log, sin, tan, tanh
from operator import attrgetter, itemgetter
from pathlib import Path
from spacy.symbols import ORTH
from warnings import warn
from contextlib import contextmanager
from fastprogress.fastprogress import MasterBar, ProgressBar
from matplotlib.patches import Patch
from pandas import Series, DataFrame

import pkg_resources
pkg_resources.require("fastprogress>=0.1.18")
from fastprogress.fastprogress import master_bar, progress_bar

#for type annotations
from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union
from types import SimpleNamespace

