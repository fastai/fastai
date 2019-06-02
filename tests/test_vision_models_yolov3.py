import pytest
from ..fastai.vision.models.yolov3 import *
import torch


@pytest.mark.parametrize("model", {"test/test_parse"})
def test_parse_cfg(model):
    blocks = parse_cfg(model)
    assert [block['type'] for block in blocks] == ['net', 'layer1', 'layer2', 'layer3']
    for block in blocks:
        if block['type'] == 'net':
            assert set(block.keys()) == {'type', 'str1', 'value1', 'value2', 'value3', 'str2'}
            assert block['str1'] == 'label1'
            assert block['value1'] == '.12'
            assert block['value2'] == '123'
            assert block['value3'] == '1.234'
            assert block['str2'] == 'label2'
        elif block['type'] == 'layer1':
            assert set(block.keys()) == {'type', 'param1', 'param2'}
            assert block['param1'] == '987'
            assert block['param2'] == 'parameter'
        elif block['type'] == 'layer2':
            assert set(block.keys()) == {'type', 'param1', 'param2'}
            assert block['param1'] == '.01'
            assert block['param2'] == 'some value'
        elif block['type'] == 'layer3':
            assert set(block.keys()) == {'type', 'param1', 'anchors'}
            assert block['param1'] == '34.56'
            assert block['anchors'] == '1,2  3,4  5,60'


@pytest.mark.parametrize("blocks", [parse_cfg('test/test_create_modules')])
def test_create_modules(blocks):
    modules, net_info = create_modules(blocks)
    nums = {'Conv2d': 0, 'BatchNorm2d': 0, 'LeakyReLU': 0, 'Upsample': 0, 'Shortcut': 0, 'Route': 0, 'Yolo': 0}
    for module in modules:
        assert type(module).__name__ == 'Sequential'
        for layer in module:
            type_ = type(layer).__name__
            nums[type_] += 1
    assert nums['Conv2d'] == 3
    assert nums['BatchNorm2d'] == 3
    assert nums['LeakyReLU'] == 2
    assert nums['Upsample'] == 1
    assert nums['Shortcut'] == 1
    assert nums['Route'] == 1
    assert nums['Yolo'] == 2


@pytest.mark.parametrize("bboxes", [torch.Tensor([[2, 2, 12, 16], [0, 0, 10, 10], [1, 0, 20, 5], [25, 1, 30, 11]])])
def test_get_IoUs(bboxes):
    result = get_IoUs(bboxes[0], bboxes[1:])
    assert round(result[0].item(), 4) == 0.3636
    assert round(result[1].item(), 4) == 0.1463
    assert result[2].item() == 0.0
    print(result)


@pytest.mark.parametrize("pretrained", {False})
def test_YOLOv3(pretrained, width=128, height=128, batch_size=2):
    model_name = 'yolov3'
    image = torch.randn([batch_size, 3, width, height])
    net = YOLOv3(model_name, pretrained=pretrained)
    detection = net(image)
    # assert detection.shape == torch.Size([batch_size, 22743, 85])
    print(len(detection))
    print(type(detection))
    print(detection[0].shape)
    print(detection[1].shape)
