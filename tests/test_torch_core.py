import pytest, torch, fastai
from fastai.gen_doc.doctest import this_tests
from fastai import *
from fastai.vision import *
from fastai.torch_core import *
from fastai.layers import *
from math import isclose

a=[1,2,3]
exp=torch.tensor(a)
b=[3,6,6]

def test_tensor_with_list():
    this_tests(tensor)
    r = tensor(a)
    assert torch.all(r==exp)

def test_tensor_with_ndarray():
    this_tests(tensor)
    b=np.array(a, dtype=np.int64)
    r = tensor(b)
    assert np_address(r.numpy()) == np_address(b)
    assert torch.all(r==exp)

def test_tensor_with_tensor():
    this_tests(tensor)
    c=torch.tensor(a)
    r = tensor(c)
    assert r.data_ptr()==c.data_ptr()
    assert torch.all(r==exp)

def test_requires_grad():
    this_tests(requires_grad)
    m = simple_cnn(b)
    assert requires_grad(m) == True

def test_requires_grad_set():
    this_tests(requires_grad)
    m = simple_cnn(b)
    requires_grad(m,False)
    allF = np.all([not p.requires_grad for p in m.parameters()])
    assert allF, "requires_grad(m,False) did not set all parameters to False"
    requires_grad(m,True)
    allT = np.all([p.requires_grad for p in m.parameters()])
    assert allT, "requires_grad(m,True) did not set all parameters to True"

def test_apply_init():
    this_tests(apply_leaf, apply_init)
    m = simple_cnn(b,bn=True)
    all2 = lambda m: nn.init.constant_(m.weight,0.2) if hasattr(m, 'weight') else m
    all7 = lambda m: nn.init.constant_(m,0.7)
    apply_leaf(m,all2)
    apply_init(m,all7)
    conv1_w = torch.full([6,3,3,3],0.7)
    bn1_w = torch.full([6],0.2)
    assert conv1_w.equal(m[0][0].weight), "Expected first colvulition layer's weights to be %r" % conv1_w
    assert bn1_w.equal(m[0][2].weight), "Expected first batch norm layers weights to be %r" % bn1_w

def test_in_channels():
    this_tests(in_channels)
    m = simple_cnn(b)
    assert in_channels(m) == 3

def test_in_channels_no_weights():
    this_tests(in_channels)
    with pytest.raises(Exception) as e_info:
        in_channels(nn.Sequential())
    assert e_info.value.args[0] == 'No weight layer'
    
def test_range_children():
    this_tests(range_children)
    m = simple_cnn(b)
    assert len(range_children(m)) == 3

def test_split_model():
    this_tests(split_model)
    m = simple_cnn(b)
    pool = split_model(m,[m[2][0]])[1][0]
    assert pool == m[2][0], "Did not properly split at adaptive pooling layer"

def test_split_no_wd_params():
    this_tests(split_no_wd_params)
    groups = split_no_wd_params(simple_cnn((1, 1, 1), bn=True))
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2

def test_set_bn_eval():
    this_tests(set_bn_eval)
    m = simple_cnn(b,bn=True)
    requires_grad(m,False)
    set_bn_eval(m)
    assert m[0][2].training == False, "Batch norm layer not properly set to eval mode"

def test_np2model_tensor():
    this_tests(np2model_tensor)
    a = np.ones([2,2])
    t = np2model_tensor(a)
    assert isinstance(t,torch.FloatTensor)
    
def test_np_address(): 
    this_tests(np_address)
    a=np.ndarray(shape=(2,2))
    add=np_address(a)
    assert isinstance(add, int)

def test_to_data():
    this_tests(to_data)    
    path = untar_data(URLs.MNIST_SAMPLE)
    data1 = ImageDataBunch.from_folder(path)
    ys1 = list(data1.y)
    a=([1,2,3],[3,6,6])
    b=([4,5,6],[4,7,7])
    data2 = torch.tensor([a,b])
    ys2= list(data2[0])
    assert isinstance(data1, fastai.vision.data.ImageDataBunch)
    assert isinstance(data1.y, ItemList)
    assert isinstance(ys1, list)
    assert isinstance(ys1[0], Category)
    assert isinstance(ys1[0].data, np.int64)
    assert isinstance(to_data(ys1[0]), np.int64)
    assert ys1[0].data == to_data(ys1[0]) 
    assert isinstance(data2, torch.Tensor)
    assert isinstance(data2[0], torch.Tensor)
    assert isinstance(ys2, list)
    assert isinstance(ys2[0], torch.Tensor)
    assert isinstance(ys2[0].data, torch.Tensor)
    assert isinstance(to_data(ys2[0]), torch.Tensor) 
    assert torch.all(torch.eq(ys2[0].data, to_data(ys2[0]))) 
    
@pytest.mark.cuda
def test_to_detach():
    this_tests(to_detach)
    a=([1.,2.,3.],[3.,6.,6.])
    b=np.array([[4,5,6],[4,7,7]])
    ta=torch.tensor(a, requires_grad=True).cuda()
    dta=to_detach(a)
    dtta=to_detach(ta, False)
    dttacpu=to_detach(ta, True)
    db=to_detach(b)
    assert ta.is_cuda
    assert isinstance(ta, (torch.cuda.FloatTensor or torch.cuda.DoubleTensor or torch.cuda.HalfTensor))
    assert ta.requires_grad 
    assert dtta.is_cuda
    assert isinstance(dtta, (torch.cuda.FloatTensor or torch.cuda.DoubleTensor or torch.cuda.HalfTensor))
    assert not dtta.requires_grad
    assert not dttacpu.is_cuda
    assert isinstance(dttacpu, (torch.FloatTensor or torch.DoubleTensor or torch.HalfTensor))
    assert not dttacpu.requires_grad
    assert isinstance(b,np.ndarray)
    assert isinstance(db,np.ndarray)
    assert np.all([b,db]) 
    
@pytest.mark.cuda    
def test_to_cpu():
    this_tests(to_cpu)
    a=([1,2,3],[3,6,6])
    b=([4,5,6],[4,7,7])
    ta=torch.tensor(a).cuda()
    tb=torch.tensor(b)
    tacpu=to_cpu(ta)
    tbcpu=to_cpu(tb)
    assert ta.is_cuda
    assert isinstance(ta, (torch.cuda.LongTensor or torch.cuda.IntTensor or torch.cuda.ShortTensor))
    assert not tacpu.is_cuda
    assert isinstance(tacpu, (torch.LongTensor or torch.IntTensor or torch.ShortTensor))
    assert not tb.is_cuda
    assert isinstance(tb, (torch.LongTensor or torch.IntTensor or torch.ShortTensor))
    assert not tbcpu.is_cuda
    assert isinstance(tbcpu, (torch.LongTensor or torch.IntTensor or torch.ShortTensor))

def test_to_half():
    this_tests(to_half)
    a=([1.,2.,3.],[3.,6.,6.])
    b=([1,2,3],[3,6,6])
    ta=torch.tensor(a)
    tb=torch.tensor(b)
    tfl=to_half(ta)
    tint=to_half(tb)
    assert tfl[0].dtype == torch.half
    assert tfl[1].dtype == torch.half
    assert tint[0].dtype == torch.int64 or torch.int32 or torch.int16
    assert tint[1].dtype == torch.int64 or torch.int32 or torch.int16
    
def test_to_float():
    this_tests(to_float)
    a=([1.,2.,3.],[3.,6.,6.])
    b=([1,2,3],[3,6,6])
    ta=torch.tensor(a)
    tb=torch.tensor(b)
    tfl=to_float(ta)
    tint=to_float(tb)
    assert tfl[0].dtype == torch.float32
    assert tfl[1].dtype == torch.float32
    assert tint[0].dtype == torch.int64 or torch.int32 or torch.int16
    assert tint[1].dtype == torch.int64 or torch.int32 or torch.int16
    
def test_children():
    this_tests(children)
    m=nn.Sequential(nn.Linear(2,2), nn.ReLU())
    ch=children(m)
    assert len(ch) == 2
    assert isinstance(ch, list)
    assert isinstance(ch[0], torch.nn.modules.linear.Linear)
    assert isinstance(ch[1], torch.nn.modules.activation.ReLU)

def test_num_children():
    this_tests(num_children)
    m=nn.Sequential(nn.Linear(2,2), nn.ReLU())
    n=num_children(m)
    assert isinstance(n, int)
    assert n == 2
    
def test_first_layer():
    this_tests(first_layer)
    m=nn.Sequential(nn.Linear(2,2), nn.ReLU())
    fl=first_layer(m)
    assert isinstance(fl, nn.Module)
    assert isinstance(fl, torch.nn.modules.linear.Linear)
    
def test_last_layer():
    this_tests(last_layer)
    m=nn.Sequential(nn.Linear(2,2), nn.ReLU())
    ll=last_layer(m)
    assert isinstance(ll, nn.Module)
    assert isinstance(ll, torch.nn.modules.activation.ReLU)
    
def test_model_type(): 
    this_tests(model_type) 
    a=np.array([1.,2.,3.]).dtype 
    b=np.array([1,2,3]).dtype 
    c=np.array(["1","2","3"]).dtype 
    assert model_type(a) == torch.float32 
    assert model_type(b) == torch.int64 
    assert model_type(c) == None   
    
def test_trange_of():
    this_tests(trange_of)
    t = trange_of(a)
    assert len(t) == len(a)
    assert t[0] == 0
    assert t[1] == 1
    assert t[2] == 2
    
def test_to_np():
    this_tests(to_np)
    a = to_np(exp)
    assert isinstance(a,np.ndarray)

def test_none_reduce_on_cpu():
    this_tests(NoneReduceOnCPU)
    y_pred = torch.ones([3,8], requires_grad=True)
    y_true = torch.zeros([3],dtype=torch.long)
    with NoneReduceOnCPU(nn.CrossEntropyLoss()) as lf:
        loss = lf(y_pred,y_true)
        assert isclose(loss.sum(),6.23,abs_tol=1e-2), "final loss does not seem to be correct"
    with NoneReduceOnCPU(F.cross_entropy) as lf:
        loss = lf(y_pred,y_true)
        assert isclose(loss.sum(),6.23,abs_tol=1e-2), "final loss without reduction does not seem to be correct"

def test_tensor_array_monkey_patch():
    this_tests('na')
    t = torch.ones(a)
    t = np.array(t)
    assert np.all(t == t), "Tensors did not properly convert to numpy arrays"
    t = torch.ones(a)
    t = np.array(t,dtype=float)
    assert np.all(t == t), "Tensors did not properly convert to numpy arrays with a dtype set"

def test_keep_parameter():
    sa = SelfAttention(128)
    this_tests(SelfAttention)
    flat = nn.Sequential(*flatten_model(sa))
    for p in sa.parameters(): assert id(p) in [id(a) for a in flat.parameters()]
