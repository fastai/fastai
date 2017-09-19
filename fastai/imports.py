from abc import abstractmethod
import collections
from collections import Iterable, Counter
from concurrent.futures import ThreadPoolExecutor
import itertools
from itertools import chain
import json
from glob import glob, iglob
import math
import os
import pickle
import random
import re
import string
import sys
import threading

import animation
import bcolz
import cv2
from ipywidgets import interact, interactive, fixed, widgets
from IPython.lib.deepreload import reload as dreload
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import PIL
import rcParams
import rc
import seaborn as sns
import scipy
import tqdm as tq
from tqdm import tqdm_notebook, tnrange

rc('animation', html='html5')
np.set_printoptions(precision=4, linewidth=100)

def in_notebook(): return 'ipykernel' in sys.modules

if in_notebook():
    def tqdm(*args, **kwargs): return tq.tqdm(*args, file=sys.stdout, **kwargs)
    def trange(*args, **kwargs): return tq.trange(*args, file=sys.stdout, **kwargs)
else:
    from tqdm import tqdm, trange
    tnrange=trange
    tqdm_notebook=tqdm
