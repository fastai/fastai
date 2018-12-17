"Utility functions to help deal with user environment"
from ..imports.torch import *
from ..core import *
from ..script import *
import fastprogress
import subprocess

def pypi_module_version_is_available(module, version):
    "Check whether module==version is available on pypi"
    # returns True/False (or None if failed to execute the check)

    # using a hack that when passing "module==" w/ no version number to pip
    # it "fails" and returns all the available versions in stderr
    try:
        cmd = f"pip install {module}=="
        result = subprocess.run(cmd.split(), shell=False, check=False,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"Error: {e}")
        return None
    else:
        if result.returncode == 1 and result.stderr:
            output = result.stderr.decode('utf-8')
            return True if version in output else False
        else:
            print(f"Some error in {cmd}")
            return None

@call_parse
def check_perf():
    "Suggest how to improve the setup to speed things up"

    from PIL import features, Image
    from packaging import version
    import pynvml

    print("Running performance checks.")

    # libjpeg_turbo check
    print("\n*** libjpeg-turbo status")
    if version.parse(Image.PILLOW_VERSION) >= version.parse("5.4.0"):
        if features.check_feature('libjpeg_turbo'):
            print("✔ libjpeg-turbo is on")
        else:
            print("✘ libjpeg-turbo is not on. It's recommended you install libjpeg-turbo to speed up JPEG decoding. See https://docs.fast.ai/performance.html#libjpeg-turbo")
    else:
        print(f"❓ libjpeg-turbo's status can't be derived - need Pillow(-SIMD)? >= 5.4.0 to tell, current version {Image.PILLOW_VERSION}")
        # XXX: remove this check/note once Pillow and Pillow-SIMD 5.4.0 is available
        pillow_ver_5_4_is_avail = pypi_module_version_is_available("Pillow", "5.4.0")
        if pillow_ver_5_4_is_avail == False:
            print("5.4.0 is not yet available, other than the dev version on github, which can be installed via pip from git+https://github.com/python-pillow/Pillow. See https://docs.fast.ai/performance.html#libjpeg-turbo")

    # Pillow-SIMD check
    print("\n*** Pillow-SIMD status")
    if re.search(r'\.post\d+', Image.PILLOW_VERSION):
        print(f"✔ Running Pillow-SIMD {Image.PILLOW_VERSION}")
    else:
        print(f"✘ Running Pillow {Image.PILLOW_VERSION}; It's recommended you install Pillow-SIMD to speed up image resizing and other operations. See https://docs.fast.ai/performance.html#pillow-simd")

    # CUDA version check
    # compatibility table: k: min nvidia ver is required for v: cuda ver
    # note: windows nvidia driver version is slightly higher, see:
    # https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
    # note: add new entries if pytorch starts supporting new cudaXX
    nvidia2cuda = {
        "410.00": "10.0",
        "384.81":  "9.0",
        "367.48":  "8.0",
    }
    print("\n*** CUDA status")
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        nvidia_ver = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
        cuda_ver   = torch.version.cuda
        max_cuda = "8.0"
        for k in sorted(nvidia2cuda.keys()):
            if version.parse(nvidia_ver) > version.parse(k): max_cuda = nvidia2cuda[k]
        if version.parse(str(max_cuda)) <= version.parse(cuda_ver):
            print(f"✔ Running the latest CUDA {cuda_ver} with NVIDIA driver {nvidia_ver}")
        else:
            print(f"✘ You are running pytorch built against cuda {cuda_ver}, your NVIDIA driver {nvidia_ver} supports cuda10. See https://pytorch.org/get-started/locally/ to install pytorch built against the faster CUDA version.")
    else:
        print(f"❓ Running cpu-only torch version, CUDA check is not relevant")

    print("\nRefer to https://docs.fast.ai/performance.html to make sense out of these checks and suggestions.")

