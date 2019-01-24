"Functions for working with the execution environment"

import os

def is_in_ipython():
    "Is the code running in the ipython environment (jupyter including)"

    program_name = os.path.basename(os.getenv('_', ''))

    if ('jupyter-notebook' in program_name or # jupyter-notebook
        'ipython'          in program_name or # ipython
        'JPY_PARENT_PID'   in os.environ):    # ipython-notebook
        return True
    else:
        return False
