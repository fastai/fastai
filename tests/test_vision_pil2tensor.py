import pytest
from fastai import *
from fastai.vision import *
import PIL

def test_vision_pil2tensor():
    path  = Path(__file__).parent / "data/test/images"
    files = list(Path(path).glob("**/*.*"))
    pil_passed, pil_failed = [],[]
    for f in files:
        try:
            im = PIL.Image.open(f)
            #provoke read of the file so we can isolate PIL issue separately
            b = np.asarray(im.convert("RGB"))
            pil_passed.append(f)
        except:
            pil_failed.append(f)

    pil2tensor_passed,pil2tensor_failed = [],[]
    for f in pil_passed:
        try :
            # it doesn't matter for the test if we convert "RGB" or "I"
            im = PIL.Image.open(f).convert("RGB")
            t  = pil2tensor(im,np.float)
            pil2tensor_passed.append(f)
        except:
            pil2tensor_failed.append(f)
            print(f"converting file: {f}  had Unexpected error:", sys.exc_info()[0])

    if len(pil2tensor_failed)>0 :
        print("\npil2tensor failed to convert the following images:")
        [print(f) for f in pil2tensor_failed]

    assert(len(pil2tensor_passed) == len(pil_passed))

def test_vision_pil2tensor_16bit():
    f    = Path(__file__) .parent/ "data/test/images/gray_16bit.png"
    im   = PIL.Image.open(f).convert("I") # so that the 16bit values are preserved as integers
    vmax = pil2tensor(im,np.int).data.numpy().max()
    assert(vmax>255)

def test_vision_pil2tensor_numpy():
    "assert that the two arrays contains the same values"
    arr  = np.random.rand(16,16,3)
    diff = np.sort( pil2tensor(arr,np.float).data.numpy().flatten() ) - np.sort(arr.flatten())
    assert( np.sum(diff==0)==len(arr.flatten()) )
