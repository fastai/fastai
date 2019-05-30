import pytest,torch
from fastai.basics import have_min_pkg_version
from fastai.gen_doc.doctest import this_tests
from fastai.text.models.qrnn import ForgetMultGPU, BwdForgetMultGPU, forget_mult_CPU, QRNN, QRNNLayer

@pytest.mark.cuda
@pytest.mark.cpp
def test_forget_mult_forward_gpu():
    this_tests(ForgetMultGPU)
    dtype = torch.double
    x,f,h,expected = range(3,8),[0.5]*5,1,range(1,7)
    x,f,expected = [torch.tensor(t, dtype=dtype)[None,:,None].cuda() for t in (x,f,expected)]
    #output = torch.zeros(1,6,1, dtype=dtype).cuda()
    #output[0,0,0] = h
    output = ForgetMultGPU.apply(x, f, h, True)
    assert torch.allclose(output,expected[:,1:])

def random_inputs(shape, batch_first, **opts):
    x = torch.randn(shape, **opts)
    f = torch.randn(shape, **opts)
    h = torch.randn((shape[0 if batch_first else 1], shape[2]), **opts)
    return x, f, h

def detach_and_clone(t):
    return t.detach().clone().requires_grad_(True)

@pytest.mark.cuda
@pytest.mark.cpp
def test_forget_mult_cuda():
    this_tests(ForgetMultGPU, BwdForgetMultGPU)
    x,f = torch.randn(5,3,20).cuda().chunk(2, dim=2)
    x,f = x.contiguous().requires_grad_(True),f.contiguous().requires_grad_(True)
    th_x,th_f = detach_and_clone(x),detach_and_clone(f)
    for (bf, bw) in [(True,True), (False,True), (True,False), (False,False)]:
        forget_mult = BwdForgetMultGPU if bw else ForgetMultGPU
        th_out = forget_mult_CPU(th_x, th_f, hidden_init=None, batch_first=bf, backward=bw)
        th_loss = th_out.pow(2).mean()
        th_loss.backward()
        out = forget_mult.apply(x, f, None, bf)
        loss = out.pow(2).mean()
        loss.backward()
        assert torch.allclose(th_out,out, rtol=1e-4, atol=1e-5)
        assert torch.allclose(th_x.grad,x.grad, rtol=1e-4, atol=1e-5)
        assert torch.allclose(th_f.grad,f.grad, rtol=1e-4, atol=1e-5)
        for p in [x,f, th_x, th_f]:
            p = p.detach()
            p.grad = None
        h = torch.randn((5 if bf else 3), 10).cuda().requires_grad_(True)
        th_h = detach_and_clone(h)
        th_out = forget_mult_CPU(th_x, th_f, hidden_init=th_h, batch_first=bf, backward=bw)
        th_loss = th_out.pow(2).mean()
        th_loss.backward()
        out = forget_mult.apply(x.contiguous(), f.contiguous(), h, bf)
        loss = out.pow(2).mean()
        loss.backward()
        assert torch.allclose(th_out,out, rtol=1e-4, atol=1e-5)
        assert torch.allclose(th_x.grad,x.grad, rtol=1e-4, atol=1e-5)
        assert torch.allclose(th_f.grad,f.grad, rtol=1e-4, atol=1e-5)
        assert torch.allclose(th_h.grad,h.grad, rtol=1e-4, atol=1e-5)
        for p in [x,f, th_x, th_f]:
            p = p.detach()
            p.grad = None

def manual_forget_mult(x, f, h=None, batch_first=True, backward=False):
    if batch_first: x,f = x.transpose(0,1),f.transpose(0,1)
    out = torch.zeros_like(x)
    prev = h if h is not None else torch.zeros_like(out[0])
    idx_range = range(x.shape[0]-1,-1,-1) if backward else range(x.shape[0])
    for i in idx_range:
        out[i] = f[i] * x[i] + (1-f[i]) * prev
        prev = out[i]
    if batch_first: out = out.transpose(0,1)
    return out

def test_forget_mult():
    this_tests(forget_mult_CPU)
    x,f = torch.randn(5,3,20).chunk(2, dim=2)
    for (bf, bw) in [(True,True), (False,True), (True,False), (False,False)]:
        th_out = manual_forget_mult(x, f, batch_first=bf, backward=bw)
        out = forget_mult_CPU(x, f, batch_first=bf, backward=bw)
        assert torch.allclose(th_out,out)
        h = torch.randn((5 if bf else 3), 10)
        th_out = manual_forget_mult(x, f, h=h, batch_first=bf, backward=bw)
        out = forget_mult_CPU(x, f, hidden_init=h, batch_first=bf, backward=bw)
        assert torch.allclose(th_out,out)

# bug in pytorch=1.0.1, fixed in 1.0.2 https://github.com/pytorch/pytorch/issues/18189
@pytest.mark.skipif(not have_min_pkg_version("torch", "1.0.2"), reason="requires torch>=1.0.2")
def test_qrnn_layer():
    this_tests(QRNNLayer)
    qrnn_fwd = QRNNLayer(10, 20, save_prev_x=True, zoneout=0, window=2, output_gate=True)
    qrnn_bwd = QRNNLayer(10, 20, save_prev_x=True, zoneout=0, window=2, output_gate=True, backward=True)
    qrnn_bwd.load_state_dict(qrnn_fwd.state_dict())
    x_fwd = torch.randn(7,5,10)
    x_bwd = x_fwd.clone().flip(1)
    y_fwd,h_fwd = qrnn_fwd(x_fwd)
    y_bwd,h_bwd = qrnn_bwd(x_bwd)
    assert torch.allclose(y_fwd, y_bwd.flip(1), rtol=1e-4, atol=1e-5)
    assert torch.allclose(h_fwd, h_bwd, rtol=1e-4, atol=1e-5)
    y_fwd,h_fwd = qrnn_fwd(x_fwd, h_fwd)
    y_bwd,h_bwd = qrnn_bwd(x_bwd, h_bwd)
    assert torch.allclose(y_fwd, y_bwd.flip(1), rtol=1e-4, atol=1e-5)
    assert torch.allclose(h_fwd, h_bwd, rtol=1e-4, atol=1e-5)

def test_qrnn_bidir():
    this_tests(QRNN)
    qrnn = QRNN(10, 20, 2, bidirectional=True, batch_first=True, window=2, output_gate=False)
    x = torch.randn(7,5,10)
    y,h = qrnn(x)
    assert y.size() == torch.Size([7, 5, 40])
    assert h.size() == torch.Size([4, 7, 20])
    #Without an out gate, the last timestamp in the forward output is the second to last hidden
    #and the first timestamp of the backward output is the last hidden
    assert torch.allclose(y[:,-1,:20], h[2])
    assert torch.allclose(y[:,0,20:], h[3])
