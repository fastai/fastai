import pytest, torch, fastai
from fastai.torch_core import *
from fastai.layers import *
from math import isclose

a=[1,2,3]
exp=torch.tensor(a)
b=[3,6,6]

def test_tensor_with_list():
    r = tensor(a)
    assert torch.all(r==exp)

def test_tensor_with_ndarray():
    b=np.array(a)
    r = tensor(b)
    assert np_address(r.numpy()) == np_address(b)
    assert torch.all(r==exp)

def test_tensor_with_tensor():
    c=torch.tensor(a)
    r = tensor(c)
    assert r.data_ptr()==c.data_ptr()
    assert torch.all(r==exp)

def test_requires_grad():
    m = simple_cnn(b)
    assert requires_grad(m) == True

def test_requires_grad_set():
    m = simple_cnn(b)
    requires_grad(m,False)
    allF = np.all([not p.requires_grad for p in m.parameters()])
    assert allF, "requires_grad(m,False) did not set all parameters to False"
    requires_grad(m,True)
    allT = np.all([p.requires_grad for p in m.parameters()])
    assert allT, "requires_grad(m,True) did not set all parameters to True"

def test_apply_init():
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
    m = simple_cnn(b)
    assert in_channels(m) == 3

def test_in_channels_no_weights():
    with pytest.raises(Exception) as e_info:
        in_channels(nn.Sequential())
    assert e_info.value.args[0] == 'No weight layer'

def test_range_children():
    m = simple_cnn(b)
    assert len(range_children(m)) == 3
    
def test_split_model():
    m = simple_cnn(b)
    pool = split_model(m,[m[2][0]])[1][0]
    assert pool == m[2][0], "Did not properly split at adaptive pooling layer"

def test_split_bn_bias():
    bn_group = split_bn_bias(simple_cnn((1, 1, 1), bn=True))[1]
    assert len(bn_group) > 0 and isinstance(bn_group[0], bn_types)

def test_set_bn_eval():
    m = simple_cnn(b,bn=True)
    requires_grad(m,False)
    set_bn_eval(m)
    assert m[0][2].training == False, "Batch norm layer not properly set to eval mode"

def test_np2model_tensor():
    a = np.ones([2,2])
    t = np2model_tensor(a)
    assert isinstance(t,torch.FloatTensor)

def test_calc_loss():
    y_pred = torch.ones([3,8], requires_grad=True)
    y_true = torch.zeros([3],dtype=torch.long)
    loss = nn.CrossEntropyLoss()
    loss = calc_loss(y_pred,y_true,loss)
    assert isclose(loss.sum(),6.23,abs_tol=1e-2), "final loss does not seem to be correct"
    loss = F.cross_entropy
    loss = calc_loss(y_pred,y_true,loss)
    assert isclose(loss.sum(),6.23,abs_tol=1e-2), "final loss without reduction does not seem to be correct"

def test_tensor_array_monkey_patch():
    t = torch.ones(a)
    t = np.array(t)
    assert np.all(t == t), "Tensors did not properly convert to numpy arrays"
    t = torch.ones(a)
    t = np.array(t,dtype=float)
    assert np.all(t == t), "Tensors did not properly convert to numpy arrays with a dtype set"
