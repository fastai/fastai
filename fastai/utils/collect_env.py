"Utility functions to help deal with user environment"
from ..imports.torch import *
from ..core import *

__all__ = ['show_install']

def get_env(name):
    "Return env var value if it's defined and not an empty string, or return Unknown"
    if name in os.environ and len(os.environ[name]):
        return os.environ[name]
    else:
        return "Unknown"

def show_install(show_nvidia_smi:bool=False):
    "Print user's setup information: python -c 'import fastai; fastai.show_install()'"

    import platform, fastai.version, subprocess

    rep = []
    opt_mods = []

    rep.append(["=== Software ===", None])

    rep.append(["python version", platform.python_version()])
    rep.append(["fastai version", fastai.__version__])
    rep.append(["torch version",  torch.__version__])

    # nvidia-smi
    cmd = "nvidia-smi"
    have_nvidia_smi = False
    try:
        result = subprocess.run(cmd.split(), shell=False, check=False, stdout=subprocess.PIPE)
    except:
        pass
    else:
        if result.returncode == 0 and result.stdout:
            have_nvidia_smi = True

    # XXX: if nvidia-smi is not available, another check could be:
    # /proc/driver/nvidia/version on most systems, since it's the
    # currently active version

    if have_nvidia_smi:
        smi = result.stdout.decode('utf-8')
        # matching: "Driver Version: 396.44"
        match = re.findall(r'Driver Version: +(\d+\.\d+)', smi)
        if match: rep.append(["nvidia driver", match[0]])

    rep.append(["torch cuda ver", torch.version.cuda])
    rep.append(["torch cuda is",
                "available" if torch.cuda.is_available() else "**Not available** "])

    # no point reporting on cudnn if cuda is not available, as it
    # seems to be enabled at times even on cpu-only setups
    if torch.cuda.is_available():
        rep.append(["torch cudnn ver", torch.backends.cudnn.version()])
        rep.append(["torch cudnn is",
                "enabled" if torch.backends.cudnn.enabled else "**Not enabled** "])

    rep.append(["\n=== Hardware ===", None])

    # it's possible that torch might not see what nvidia-smi sees?
    gpu_total_mem = []
    nvidia_gpu_cnt = 0
    if have_nvidia_smi:
        try:
            cmd = "nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader"
            result = subprocess.run(cmd.split(), shell=False, check=False, stdout=subprocess.PIPE)
        except:
            print("have nvidia-smi, but failed to query it")
        else:
            if result.returncode == 0 and result.stdout:
                output = result.stdout.decode('utf-8')
                gpu_total_mem = [int(x) for x in output.strip().split('\n')]
                nvidia_gpu_cnt = len(gpu_total_mem)


    if nvidia_gpu_cnt: rep.append(["nvidia gpus", nvidia_gpu_cnt])

    torch_gpu_cnt = torch.cuda.device_count()
    if torch_gpu_cnt:
        rep.append(["torch available", torch_gpu_cnt])
        # information for each gpu
        for i in range(torch_gpu_cnt):
            rep.append([f"  - gpu{i}", (f"{gpu_total_mem[i]}MB | " if gpu_total_mem else "") + torch.cuda.get_device_name(i)])
    else:
        if nvidia_gpu_cnt:
            rep.append([f"Have {nvidia_gpu_cnt} GPU(s), but torch can't use them (check nvidia driver)", None])
        else:
            rep.append([f"No GPUs available", None])


    rep.append(["\n=== Environment ===", None])

    rep.append(["platform", platform.platform()])

    if platform.system() == 'Linux':
        try:
            import distro
        except ImportError:
            opt_mods.append('distro');
            # partial distro info
            rep.append(["distro", platform.uname().version])
        else:
            # full distro info
            rep.append(["distro", ' '.join(distro.linux_distribution())])

    rep.append(["conda env", get_env('CONDA_DEFAULT_ENV')])
    rep.append(["python", sys.executable])
    rep.append(["sys.path", "\n".join(sys.path)])

    print("\n\n```text")

    keylen = max([len(e[0]) for e in rep if e[1] is not None])
    for e in rep:
        print(f"{e[0]:{keylen}}", (f": {e[1]}" if e[1] is not None else ""))

    if have_nvidia_smi:
        if show_nvidia_smi == True: print(f"\n{smi}")
    else:
        if torch_gpu_cnt:
            # have gpu, but no nvidia-smi
            print("no nvidia-smi is found")
        else:
            print("no supported gpus found on this system")

    print("```\n")

    print("Please make sure to include opening/closing ``` when you paste into forums/github to make the reports appear formatted as code sections.\n")

    if opt_mods:
        print("Optional package(s) to enhance the diagnostics can be installed with:")
        print(f"pip install {' '.join(opt_mods)}")
        print("Once installed, re-run this utility to get the additional information")
