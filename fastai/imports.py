from IPython.lib.deepreload import reload as dreload
import PIL, os, numpy as np, math, collections, threading, json, bcolz, random, scipy, cv2
import random, pandas as pd, pickle, sys, itertools, string, sys, re, datetime, time
import seaborn as sns, matplotlib
import IPython, graphviz, sklearn_pandas, sklearn, warnings
from abc import abstractmethod
from glob import glob, iglob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import chain
from functools import partial
from collections import Iterable, Counter, OrderedDict
from isoweek import Week
from pandas_summary import DataFrameSummary
from IPython.lib.display import FileLink
from PIL import Image, ImageEnhance, ImageOps
from sklearn import metrics, ensemble, preprocessing

from matplotlib import pyplot as plt, rcParams, animation
from ipywidgets import interact, interactive, fixed, widgets
matplotlib.rc('animation', html='html5')
np.set_printoptions(precision=5, linewidth=110, suppress=True)

def in_notebook(): return 'ipykernel' in sys.modules

import tqdm as tq
from tqdm import tqdm_notebook, tnrange
if in_notebook():
    def tqdm(*args, **kwargs): return tq.tqdm(*args, file=sys.stdout, **kwargs)
    def trange(*args, **kwargs): return tq.trange(*args, file=sys.stdout, **kwargs)
else:
    from tqdm import tqdm, trange
    tnrange=trange
    tqdm_notebook=tqdm

#cv2.setNumThreads(0)

