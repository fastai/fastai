import pytest,torch
from fastai.text.models import qrnn

@pytest.mark.cuda
def test_forget_mult_forward_gpu():
    dtype = torch.double
    x,f,h,expected = range(3,8),[0.5]*5,1,range(1,7)
    x,f,expected = [torch.tensor(t, dtype=dtype)[None,:,None].cuda() for t in (x,f,expected)]
    output = torch.zeros(1,6,1, dtype=dtype).cuda()
    output[0,0,0] = h
    qrnn.forget_mult_cuda.forward(x, f, output, True)
    assert torch.allclose(output,expected)

def random_inputs(shape, batch_first, **opts):
    x = torch.randn(shape, **opts)
    f = torch.randn(shape, **opts)
    h = torch.randn((shape[0 if batch_first else 1], shape[2]), **opts)
    return x, f, h

@pytest.mark.cuda
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("shape", [(1, 1, 1), (7, 11, 13)])
def test_compare_forget_mult_forward_implementations(shape, batch_first):
    x, f, h = random_inputs(shape, batch_first, dtype=torch.double)
    forget_cpu = qrnn.ForgetMultCPU(batch_first)
    output_cpu = forget_cpu(x, f, h)

    forget_gpu = qrnn.ForgetMultGPU(batch_first)
    output_gpu = forget_gpu(x.cuda(), f.cuda(), h.cuda()).cpu()
    assert torch.allclose(output_cpu,output_gpu)
