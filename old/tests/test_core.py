import pytest, torch, unittest, bcolz
import numpy as np
from fastai import core
from unittest import mock
from unittest.mock import Mock
from testfixtures import tempdir

def test_sum_geom():
    assert core.sum_geom(1, 1, 1) == 1
    assert core.sum_geom(1, 1, 3) == 3
    assert core.sum_geom(3, 10, 4) == 3333
    assert core.sum_geom(0, 2, 3) == 0
    assert core.sum_geom(1, 0, 3) == 1
    assert core.sum_geom(1, 2, 0) == 0

def test_map_none():
  def fn(x): return x
  assert core.map_none(None, fn) == None
  assert core.map_none("not none", fn) == "not none"

def test_delistify():
  assert core.delistify([1]) == 1
  assert core.delistify((1)) == 1
  assert core.delistify("non list") == "non list"
  assert core.delistify(object) == object

  with pytest.raises(IndexError):
    assert core.delistify([])

def test_datafy():
  x = Mock(data={})
  assert core.datafy(x) == {}
  assert core.datafy([x]) == [{}]
  assert core.datafy([x, x]) == [{}, {}]

@mock.patch("fastai.core.to_gpu", lambda x, *args, **kwargs: x)
def test_T():
  tensor = torch.ones([1, 2])
  np.testing.assert_equal(core.to_np(core.T(tensor)), [[1, 1]])

  array = np.arange(0, 5)
  assert core.T(array.astype(np.int)).type() == "torch.LongTensor"
  assert core.T(array.astype(np.float)).type() == "torch.FloatTensor"

  with mock.patch("fastai.core.to_half") as to_half_mock:
    core.T(array.astype(np.float), half=True)
    to_half_mock.assert_called_once()

  with pytest.raises(NotImplementedError):
    assert core.T(array.astype(np.object))

def test_create_variable_passing_Variable_object():
  v = torch.autograd.Variable(core.T(np.arange(0, 3)))
  cv = core.create_variable(v, volatile=True)
  if core.IS_TORCH_04: assert (cv == v).all()
  else: assert cv is v

@mock.patch("fastai.core.Variable")
def test_create_variable(VariableMock):
  v = np.arange(0, 3)

  with mock.patch("fastai.core.IS_TORCH_04", True):
    core.create_variable(v, volatile=True)
    assert VariableMock.call_args[1] == {"requires_grad": False}
  
  with mock.patch("fastai.core.IS_TORCH_04", False):
    core.create_variable(v, volatile=True)
    assert VariableMock.call_args[1] == {"requires_grad": False, "volatile": True}

@mock.patch("fastai.core.create_variable")
def test_V_(create_variable_mock):
  core.V_("foo")

  create_variable_mock.assert_called_with('foo', requires_grad=False, volatile=False)

@mock.patch("fastai.core.map_over")
def test_V(map_over_mock):
  core.V("foo")

  assert map_over_mock.call_args[0][0] == 'foo'
  assert type(map_over_mock.call_args[0][1]) == type(lambda:0)

def test_to_np():
  array = np.arange(0, 3).astype(np.float)
  assert core.to_np(array) is array

  tensor = core.T(array)
  result = core.to_np([tensor, tensor])
  np.testing.assert_equal(result[0], array)
  np.testing.assert_equal(result[1], array)

  variable = core.V(array)
  np.testing.assert_equal(core.to_np(variable), array)

  with mock.patch("torch.cuda.is_available") as is_available_mock:
    with mock.patch("fastai.core.is_half_tensor") as is_half_tensor_mock:
      is_available_mock.return_value=True
      is_half_tensor_mock.return_value=True

      tensor = core.T(array.astype(np.int))

      array = core.to_np(tensor)
      np.testing.assert_equal(array, [0., 1., 2.])
      assert array.dtype in (np.float32, np.float64)


def test_noop():
  assert core.noop() is None

def test_chain_params():
  modules = [torch.nn.Linear(3, 2), torch.nn.Linear(2, 1)]

  params = core.chain_params(modules)
  assert len(params) == 4
  assert list(params[0].size()) == [2, 3]
  assert list(params[1].size()) == [2]
  assert list(params[2].size()) == [1, 2]
  assert list(params[3].size()) == [1]

  params = core.chain_params(torch.nn.Linear(2, 2))
  assert list(params[0].size()) == [2, 2]
  assert list(params[1].size()) == [2]

def test_set_trainable_attr():
  linear = torch.nn.Linear(2, 1)
  core.set_trainable_attr(linear, False)

  assert linear.trainable == False
  for param in linear.parameters():
    assert param.requires_grad == False

def test_apply_leaf():
  spy = Mock(name="apply_leaf_spy")
  fn = lambda x: spy(x)
  layer1 = torch.nn.Linear(2, 2)
  layer2 = torch.nn.Linear(2, 1)
  model = torch.nn.Sequential(layer1, layer2)

  core.apply_leaf(model, fn)

  assert spy.call_count == 3
  assert spy.call_args_list[0][0][0] is model
  assert spy.call_args_list[1][0][0] is layer1
  assert spy.call_args_list[2][0][0] is layer2

def test_set_trainable():
  layer1 = torch.nn.Linear(2, 2)
  layer2 = torch.nn.Linear(2, 1)
  model = torch.nn.Sequential(layer1, layer2)

  params_require_grad_before = list(filter(lambda param: param.requires_grad == True,
                                    model.parameters()))

  core.set_trainable(model, False)

  params_require_grad_after = list(filter(lambda param: param.requires_grad == True,
                                    model.parameters()))

  assert len(params_require_grad_before) == 4
  assert len(params_require_grad_after) == 0

  assert model.trainable == False
  assert layer1.trainable == False
  assert layer2.trainable == False

@mock.patch("fastai.core.optim.SGD")
def test_SGD_Momentum(sgd_mock):
  sgd = core.SGD_Momentum(0.2)
  sgd("foo", param1=1, param2=2)

  sgd_mock.assert_called_with('foo', momentum=0.2, param1=1, param2=2)

def test_one_hot():
  labels = [0, 1, 0, 2, 0, 3]
  num_classes = 4
  one_hot = core.one_hot(labels, num_classes)

  np.testing.assert_equal(one_hot, [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
  ])

@mock.patch("fastai.core.num_cpus")
def test_partition_by_cores(num_cpus_mock):
  x = [0, 1, 2, 3, 4]

  num_cpus_mock.return_value = 1
  assert core.partition_by_cores(x) == [[0, 1, 2, 3, 4]]

  num_cpus_mock.return_value = 2
  assert core.partition_by_cores(x) == [[0, 1, 2], [3, 4]]

  num_cpus_mock.return_value = 3
  assert core.partition_by_cores(x) == [[0, 1], [2, 3], [4]]

  num_cpus_mock.return_value = 4
  assert core.partition_by_cores(x) == [[0, 1], [2, 3], [4]]

@mock.patch("fastai.core.os")
def test_num_cpus_with_sched_getaffinity(os_mock):
  os_mock.sched_getaffinity = Mock(return_value=["foo", "bar"])

  assert core.num_cpus() == 2

@mock.patch("fastai.core.os")
def test_num_cpus_without_sched_getaffinity(os_mock):
  os_mock.sched_getaffinity = Mock(side_effect=AttributeError)
  os_mock.cpu_count = Mock(return_value=3)

  assert core.num_cpus() == 3

def test_partition_functionality():

  def test_partition(a, sz, ex):
    result = core.partition(a, sz)
    assert len(result) == len(ex)
    assert all([a == b for a, b in zip(result, ex)])

  a = [1,2,3,4,5]
  
  sz = 2
  ex = [[1,2],[3,4],[5]]
  test_partition(a, sz, ex)

  sz = 3
  ex = [[1,2,3],[4,5]]
  test_partition(a, sz, ex)

  sz = 1
  ex = [[1],[2],[3],[4],[5]]
  test_partition(a, sz, ex)

  sz = 6
  ex = [[1,2,3,4,5]]
  test_partition(a, sz, ex)

  sz = 3
  a = []
  result = core.partition(a, sz)
  assert len(result) == 0


def test_partition_error_handling():
  sz = 0
  a = [1,2,3,4,5]
  with pytest.raises(ValueError):
    core.partition(a, sz)


def test_split_by_idxs_functionality():

  seq = [1,2,3,4,5,6]
  
  def test_split_by_idxs(seq, idxs, ex):
    test_result = []
    for item in core.split_by_idxs(seq, idxs):
      test_result.append(item)
    
    assert len(test_result) == len(ex)
    assert all([a == b for a,b in zip(test_result, ex)])
  
  idxs = [2]
  ex = [[1,2],[3,4,5,6]]

  test_split_by_idxs(seq, idxs, ex)
  
  idxs = [1,2]
  ex = [[1],[2],[3,4,5,6]]
  test_split_by_idxs(seq, idxs, ex)

  idxs = [2,4,5]
  ex = [[1,2],[3,4],[5],[6]]
  test_split_by_idxs(seq, idxs, ex)

  idxs = []
  ex = [[1,2,3,4,5,6]]
  test_split_by_idxs(seq, idxs, ex)


def test_split_by_idxs_error_handling():
  seq = [1,2,3,4]
  idxs = [5]

  gen = core.split_by_idxs(seq, idxs)
  with pytest.raises(KeyError):
    next(gen)

def test_BasicModel():
  layer_1 = torch.nn.Linear(2, 2)
  layer_2 = torch.nn.Linear(2, 1)
  model = torch.nn.Sequential(layer_1, layer_2)
  basic = core.BasicModel(model, name="foo")

  assert basic.model is model
  assert basic.name == "foo"

  layers = basic.get_layer_groups()
  assert layers == [layer_1, layer_2]

def test_SingleModel():
  layer_1 = torch.nn.Linear(2, 2)
  layer_2 = torch.nn.Linear(2, 1)
  model = torch.nn.Sequential(layer_1, layer_2)
  single_model = core.SingleModel(model, name="foo")

  assert single_model.get_layer_groups() == [model]

class TestSimpleNet(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(42)
    self.layers = [2, 3, 2]
    self.simple_net = core.SimpleNet(self.layers)
  
  def test__init__(self):
    assert isinstance(self.simple_net.layers, torch.nn.ModuleList)
    assert len(self.simple_net.layers) == 2

    assert self.simple_net.layers[0].in_features == 2
    assert self.simple_net.layers[0].out_features == 3

    assert self.simple_net.layers[1].in_features == 3
    assert self.simple_net.layers[1].out_features == 2

  @mock.patch("fastai.core.to_gpu", lambda x, *args, **kwargs: x)
  def test_forward(self):
    x = core.V(np.array([[1., 2.]]), requires_grad=False)
    output = core.to_np(self.simple_net.forward(x))

    np.testing.assert_almost_equal(output, [[-1.435481, -0.27181]], decimal=4)

@tempdir()
def test_save_load(tempdir):
  array = np.arange(0, 5)
  core.save(f"{tempdir.path}/data.pk", array)

  data = core.load(f"{tempdir.path}/data.pk")
  np.testing.assert_equal(data, [0, 1, 2, 3, 4])

@mock.patch("pickle.load")
@mock.patch("builtins.open")
def test_load2(open_mock, load_mock):
  core.load2("filename.pk")

  assert load_mock.call_args[1]['encoding'] == 'iso-8859-1'

@tempdir()
def test_load_array(tempdir):
  rootdir=tempdir.path
  bcolz.carray(np.arange(0,5), mode='w', rootdir=rootdir)

  array = core.load_array(rootdir)
  np.testing.assert_equal(array, [0, 1, 2, 3, 4])

def test_chunk_iter():
  nums = iter(range(10))

  chunks = core.chunk_iter(nums, chunk_size=3)
  assert list(chunks) == [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9]
  ]

@mock.patch("fastai.core.torch")
@mock.patch("contextlib.suppress")
def test_set_grad_enabled(suppress_mock, torch_mock):
  torch_mock.set_grad_enabled = Mock()

  with mock.patch("fastai.core.IS_TORCH_04", True):
    core.set_grad_enabled("foo")
    torch_mock.set_grad_enabled.assert_called_with("foo")

  with mock.patch("fastai.core.IS_TORCH_04", False):
    core.set_grad_enabled("foo")
    suppress_mock.assert_called_once()

@mock.patch("fastai.core.torch")
@mock.patch("contextlib.suppress")
def test_no_grad(suppress_mock, torch_mock):
  torch_mock.no_grad = Mock()

  with mock.patch("fastai.core.IS_TORCH_04", True):
    core.no_grad_context()
    torch_mock.no_grad.assert_called_once()

  with mock.patch("fastai.core.IS_TORCH_04", False):
    core.no_grad_context()
    suppress_mock.assert_called_once()

