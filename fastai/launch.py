import subprocess,torch,os,sys
from fastcore.basics import *
from fastcore.script import *

@call_parse
def main(
    gpus:Param("The GPUs to use for distributed training", str)='all',
    script:Param("Script to run", str, opt=False)='',
    args:Param("Args to pass to script", nargs='...', opt=False)=''
):
    "PyTorch distributed training launch helper that spawns multiple distributed processes"
    current_env = os.environ.copy()
    gpus = list(range(torch.cuda.device_count())) if gpus=='all' else gpus.split(',')
    current_env["WORLD_SIZE"] = str(len(gpus))
    current_env["MASTER_ADDR"] = '127.0.0.1'
    current_env["MASTER_PORT"] = '29500'

    procs = []
    for i,gpu in enumerate(gpus):
        current_env["RANK"],current_env["DEFAULT_GPU"] = str(i),str(gpu)
        procs.append(subprocess.Popen([sys.executable, "-u", script] + args, env=current_env))
    for p in procs: p.wait()

