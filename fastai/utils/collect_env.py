"Utility functions to help deal with user environment"
from ..imports.torch import *
from ..core import *

__all__ = ['show_install']

def show_install(show_nvidia_smi:bool=False):
    "Print user's setup information: python -c 'import fastai; fastai.show_install()'"

    import platform, fastai.version, subprocess

    rep = []
    rep.append(["platform", platform.platform()])

    opt_mods = []

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

    rep.append(["python", platform.python_version()])
    rep.append(["fastai", fastai.__version__])
    rep.append(["torch",  torch.__version__])

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
        if match: rep.append(["nvidia dr.", match[0]])

    # nvcc
    cmd = "nvcc --version"
    have_nvcc = False
    try:
        result = subprocess.run(cmd.split(), shell=False, check=False, stdout=subprocess.PIPE)
    except:
        pass
    else:
        if result.returncode == 0 and result.stdout:
            have_nvcc = True

    nvcc_cuda_ver = "Unknown"
    if have_nvcc:
        nvcc = result.stdout.decode('utf-8')
        # matching: "Cuda compilation tools, release 9.2, V9.2.148"
        match = re.findall(r'V(\d+\.\d+.\d+)', nvcc)
        if match: nvcc_cuda_ver = match[0]

    cuda_is_available = torch.cuda.is_available()
    if not cuda_is_available: rep.append(["torch cuda", "Not available"])

    rep.append(["torch cuda", torch.version.cuda])
    rep.append(["nvcc  cuda", nvcc_cuda_ver])

    # disable this info for now, seems to be available even on cpu-only systems
    #rep.append(["cudnn", torch.backends.cudnn.version()])
    #rep.append(["cudnn avail", torch.backends.cudnn.enabled])

    gpu_cnt = torch.cuda.device_count()
    rep.append(["torch gpus", gpu_cnt])

    # it's possible that torch might not see what nvidia-smi sees?
    gpu_total_mem = []
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

    # information for each gpu
    for i in range(gpu_cnt):
        rep.append([f"  [gpu{i}]", None])
        rep.append(["  name", torch.cuda.get_device_name(i)])
        if gpu_total_mem: rep.append(["  total mem", f"{gpu_total_mem[i]}MB"])

    print("\n\n```")

    keylen = max([len(e[0]) for e in rep])
    for e in rep:
        print(f"{e[0]:{keylen}}", (f": {e[1]}" if e[1] else ""))

    if have_nvidia_smi:
        if show_nvidia_smi == True: print(f"\n{smi}")
    else:
        if gpu_cnt:
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
