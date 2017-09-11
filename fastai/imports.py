from IPython.lib.deepreload import reload as dreload
import PIL, os, numpy as np, math, collections, cv2, threading, json, bcolz, random, scipy
import random, pandas as pd, pickle, sys, itertools, string, sys, re
import seaborn as sns
from abc import abstractmethod
from glob import glob, iglob
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from collections import Iterable, Counter

from matplotlib import pyplot as plt, rcParams, animation, rc
from ipywidgets import interact, interactive, fixed, widgets
rc('animation', html='html5')
np.set_printoptions(precision=4, linewidth=100)

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

