"""Get OS specific nvml wrapper. On OSX we use pynvx as drop in replacement for pynvml"""

import platform
from ..script import *


def load_pynvml_env():
    import pynvml # nvidia-ml-py3
    
    if platform.system() == "Darwin":
        try:
            from pynvx import pynvml
        except:
            print("please install pynvx on OSX: pip install pynvx")
            sys.exit(1)

        pynvml.nvmlInit()
        return pynvml

    pynvml.nvmlInit()

    return pynvml
