import torch
import torch.nn as nn
import numpy as np
from ...layers import *


__all__ = ['parse_cfg', 'create_modules', 'get_IoUs', 'rewrite_results', 'YOLOv3']


class NoNetBlock(Exception):

    def __init__(self, blocks):
        self.blocks = blocks

    def __str__(self):
        return repr("There is no 'net' block in given list (type of blocks: {})".format(
            ', '.join([block['type'] for block in self.blocks])))


class LayerNotDefined(Exception):

    def __init__(self, layer_name):
        self.layer = layer_name

    def __str__(self):
        return repr("There is no such type of layer - {}".format(self.layer))


def parse_cfg(model):
    """
    Reading configuration file for given model.

    Parsing file with information about network configuration into list of blocks.
    Args:
        model: name of the model - there must be file {name}.cfg in the cfg directory
    Returns:
        For cfg file like:

        [net]
        # Training
        batch=10
        [convolutional]
        size=3
        activation=leaky

        The function returns:
        blocks = [{'type': 'convolutional', 'size': '3', 'activation': 'leaky'}]
        net_info = {'type': 'net', 'batch': '10'}

        blocks: list of dicts, each dict represents a module, items are parameters,
                apart from item 'type', which is a type of the module.
        net_info: dict with general parameters of the network
    """

    blocks = []
    with open('d:/Studia/IVrok/ADPB/pycharm_fastai/data/cfg/{}.cfg'.format(model), 'r') as file:
        block = {}
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('['):
                    if block:
                        blocks.append(block)
                        block = {}
                    block['type'] = line.strip('[]')
                else:
                    key, value = [el.strip() for el in line.split('=')]
                    block[key] = value
        blocks.append(block)

    if blocks[0]['type'] == 'net':
        net_info = blocks[0]
    else:
        raise NoNetBlock(blocks)

    return blocks[1:], net_info


def create_modules(blocks, dims):
    """
    Creating network's layers based on given blocks.

    Making layers (as torch.nn.Module objects) based on list of blocks (output from parse_cfg).
    Args:
        blocks: list of dicts, each dict is description of layer based on cfg file (output from parse_cfg)
        dims: dimensions of default input image
    Returns:
        modules: (nn.ModuleList) list of modules of layers created based on given blocks
    """

    modules = nn.ModuleList([])
    filters = [3]  # three RGB input channels

    for index, block in enumerate(blocks):

        module = nn.Sequential()
        t = block['type']

        if t == 'convolutional':

            # required parameters
            in_channels = filters[-1]
            out_channels = int(block['filters'])
            kernel_size = int(block['size'])

            # default parameters
            batch_norm = False
            bias = True
            padding = 0
            try:
                if block['batch_normalize'] == '1':
                    batch_norm = True
                    bias = False  # not needed because of beta parameter in batch normalization
            except KeyError:
                pass
            activation = block['activation']
            stride = int(block['stride'])
            try:
                if block['pad'] == '1':
                    padding = (kernel_size - 1) // 2  # number of pixels needed from one side to use kernel
            except KeyError:
                pass

            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
            module.add_module("conv_{}".format(index), conv)

            if batch_norm:
                bn = nn.BatchNorm2d(out_channels)
                module.add_module("batch_norm_{}".format(index), bn)
            if activation == 'leaky':
                activ = nn.LeakyReLU(0.01, inplace=False)  # default values of parameters
                module.add_module("leaky_{}".format(index), activ)
            elif activation == 'linear':
                pass
            else:
                raise LayerNotDefined('{} activation'.format(activation))

        elif t == 'upsample':
            scale_factor = int(block['stride'])
            upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')  # mode defined by the authors of YOLOv3
            module.add_module("upsample_{}".format(index), upsample)
            out_channels = filters[-1]

        elif t == 'route':
            layers = []
            for l in list(map(int, block['layers'].split(','))):
                if l < 0:
                    layers.append(index + l + 1)
                else:
                    layers.append(l + 1)  # +1 becuase first element of outpus is input data
            route = Route(layers)
            module.add_module("route_{}".format(index), route)
            out_channels = sum([filters[j] for j in layers])

        elif t == 'shortcut':
            activation = block['activation']
            from_layer, prev_layer = index + int(block['from']) + 1, index  # +1 because first element of outputs is input data
            if activation != 'linear':
                raise LayerNotDefined('shortcut with {} activation'.format(activation))
            shortcut = Shortcut(from_layer, prev_layer)
            module.add_module("shortcut_{}".format(index), shortcut)
            out_channels = filters[-1]

        elif t == 'yolo':
            mask = [int(el) for el in block['mask'].split(',')]
            anchors = block['anchors'].split('  ')
            assert len(anchors) == int(block['num'])
            anchors = [tuple(map(int, el.strip(',').split(','))) for i, el in enumerate(anchors) if i in mask]
            yolo = Yolo(anchors, dims, int(block['classes']))
            module.add_module("yolo_{}".format(index), yolo)
            out_channels = -1  # designation of output from yolo layer

        else:
            raise LayerNotDefined(t)

        filters.append(out_channels)
        modules.append(module)

    return modules


def get_corners(center_x, center_y, width, height):
    """
    Transform coordinates of the center, width and height of the bounding
    box into coordinates of the top-left and right-bottom corners.
    """
    x1 = center_x - width/2
    y1 = center_y - height/2
    x2 = center_x + width/2
    y2 = center_y + height/2

    return x1, y1, x2, y2


def get_IoUs(bbox, others):
    """
    Counting Intersection over Union value for given bounding box with each of boxes in given 'others' list.
    :param bbox: bounding box for which IoU should be counted
    :param others: list of bounding boxes to compare with bbox
    :return: list of IoUs values for each bounding box
    """

    x1, y1, x2, y2 = get_corners(*bbox[:4])
    ious = torch.zeros(others.size(0))
    for i, other in enumerate(others):
        xx1, yy1, xx2, yy2 = get_corners(*other[:4])
        # intersection area
        if xx1 > x2 or xx2 < x1 or yy1 > y2 or yy2 < y1:
            iou = 0.0
        else:
            inter_x1, inter_y1 = torch.max(x1, xx1), torch.max(y1, yy1)
            inter_x2, inter_y2 = torch.min(x2, xx2), torch.min(y2, yy2)
            inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))

            # union area
            union_area = (x2 - x1) * (y2 - y1) + (xx2 - xx1) * (yy2 - yy1) - inter_area

            iou = inter_area / union_area

        ious[i] = iou

    return ious


def rewrite_results(detections, confidence, nms_conf):

    corners = torch.zeros(detections.size(0), detections.size(1), 4)
    corners[:, :, 0] = detections[:, :, 0] - detections[:, :, 2]/2
    corners[:, :, 1] = detections[:, :, 1] - detections[:, :, 3]/2
    corners[:, :, 2] = detections[:, :, 0] + detections[:, :, 2]/2
    corners[:, :, 3] = detections[:, :, 1] + detections[:, :, 3]/2
    detections[:, :, :4] = corners

    outputs = []
    max_bbox = 0
    for image in detections:
        # select bounding boxes which objectness score is above confidence score
        bboxes = image[(image[:, 4] > confidence).nonzero().squeeze(), :]
        if bboxes.size(0) == 0:
            outputs.append(None)
            continue
        class_conf, class_number = torch.max(bboxes[:, 5:], 1)
        class_conf, class_number = class_conf.float().unsqueeze(1), class_number.float().unsqueeze(1)
        bboxes = torch.cat((bboxes[:, :5], class_conf, class_number), 1)

        # get list of possible classes of bounding boxes from the image
        img_classes = torch.from_numpy(np.unique(bboxes[:, -1].detach().numpy()))

        # perform Non-maximum suppression
        selected_bboxes = []
        for class_ in img_classes:
            # get bounding boxes indices for given class
            class_bboxes = [index for index, bbox in enumerate(bboxes) if bbox[-1] == class_]
            if len(class_bboxes) == 0:
                continue
            # sort bounding boxes descending by the objectness confidence
            class_bboxes = [class_bboxes[i] for i in bboxes[class_bboxes, 4].sort(descending=True)[1]]

            removed = []
            for index in class_bboxes:
                if index not in removed:
                    next_indices = [i for i in class_bboxes[index+1:] if i not in removed]

                    ious = get_IoUs(bboxes[index], bboxes[next_indices])

                    removed += [i for ii, i in enumerate(next_indices) if ious[ii] >= nms_conf]

            selected_bboxes += [i for i in class_bboxes if i not in removed]

        if len(selected_bboxes) > max_bbox:
            max_bbox = len(selected_bboxes)
        outputs.append(bboxes[selected_bboxes])

    batch_size = detections.size(0)
    assert len(outputs) == batch_size
    output_coords = torch.zeros(batch_size, max_bbox, 4)
    output_classes = torch.zeros(batch_size, max_bbox)

    for i, output in enumerate(outputs):
        if output is not None:
            start_pos = max_bbox - len(output)
            output_coords[i, start_pos:] = output[:, :4]
            output_classes[i, start_pos:] = output[:, -1]

    return [output_coords, output_classes]


def yolo_loss(input, target, lambda_coords, lambda_noobj):

    # width, height and center coords of the target
    w_ = target[:, :, 2] - target[:, :, 0]
    h_ = target[:, :, 3] - target[:, :, 1]
    x_ = target[:, :, 0] + w_/2
    y_ = target[:, :, 1] + h_/2


    pass


class YOLOv3(nn.Module):

    def __init__(self, model_name='yolov3', pretrained=False):
        super(YOLOv3, self).__init__()
        self.blocks, self.net_info = parse_cfg(model_name)
        self.inp_dims = torch.Size([int(self.net_info['width']), int(self.net_info['height'])])
        self.modules_list = create_modules(self.blocks, self.inp_dims)
        if pretrained:
            self.header = self.load_weights(model_name)
            self.seen = self.header[3]

    def forward(self, data):
        assert data.size(2) == data.size(3)  # YOLOv3 architecture accepts only square images
        outputs = [data]
        for module in self.modules_list:
            type_ = type(module[0]).__name__
            if type_ in ['Shortcut', 'Route']:
                output = module(outputs)
            else:
                output = module(outputs[-1])
            if type_ == 'Yolo':
                if 'detections' not in locals():
                    detections = output
                else:
                    detections = torch.cat((detections, output), 1)
            outputs.append(output)

        return detections

    def load_weights(self, model_name):
        file = open('d:/Studia/IVrok/ADPB/weights/{}.weights'.format(model_name), 'rb')
        header = torch.from_numpy(np.fromfile(file, dtype=np.int32, count=5))
        for i, module in enumerate(self.modules_list):

            if self.blocks[i]['type'] == 'convolutional':

                conv = module[0]

                batch_norm = False
                try:
                    if self.blocks[i]['batch_normalize'] == '1':
                        batch_norm = True
                except KeyError:
                    pass

                if batch_norm:

                    bn = module[1]
                    bn_biases = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=bn.bias.numel())).\
                        view_as(bn.bias.data)
                    bn_weights = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=bn.weight.numel())).\
                        view_as(bn.weight.data)
                    bn_running_mean = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=bn.running_mean.numel())).\
                        view_as(bn.running_mean)
                    bn_running_var = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=bn.running_var.numel())).\
                        view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:

                    conv_biases = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=conv.bias.numel())).\
                        view_as(conv.bias.data)

                    conv.bias.data.copy_(conv_biases)

                conv_weights = torch.from_numpy(np.fromfile(file, dtype=np.float32, count=conv.weight.numel())).\
                    view_as(conv.weight.data)

                conv.weight.data.copy_(conv_weights)

        file.close()
        return header


