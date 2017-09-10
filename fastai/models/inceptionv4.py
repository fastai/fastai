import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import sys

model_urls = {
    'imagenet': 'http://webia.lip6.fr/~cadene/Downloads/inceptionv4-97ef9c30.pth'
}

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out

class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out

class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out

class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2)
        )
        
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(), # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classif = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x) 
        return x
    
def inceptionv4(pretrained=True):
    model = InceptionV4()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['imagenet']))
    return model

######################################################################
## Load parameters from HDF5 to Dict
######################################################################

def load_conv2d(state_dict, name_pth, name_tf):
    h5f = h5py.File('dump/InceptionV4/'+name_tf+'.h5', 'r')
    state_dict[name_pth+'.conv.weight'] = torch.from_numpy(h5f['weights'][()]).permute(3, 2, 0, 1)
    out_planes = state_dict[name_pth+'.conv.weight'].size(0)
    state_dict[name_pth+'.bn.weight'] = torch.ones(out_planes)
    state_dict[name_pth+'.bn.bias'] = torch.from_numpy(h5f['beta'][()])
    state_dict[name_pth+'.bn.running_mean'] = torch.from_numpy(h5f['mean'][()])
    state_dict[name_pth+'.bn.running_var'] = torch.from_numpy(h5f['var'][()])
    h5f.close()

def load_linear(state_dict, name_pth, name_tf):
    h5f = h5py.File('dump/InceptionV4/'+name_tf+'.h5', 'r')
    state_dict[name_pth+'.weight'] = torch.from_numpy(h5f['weights'][()]).t()
    state_dict[name_pth+'.bias'] = torch.from_numpy(h5f['biases'][()])
    h5f.close()

def load_mixed_4a_7a(state_dict, name_pth, name_tf):
    load_conv2d(state_dict, name_pth+'.branch0.0', name_tf+'/Branch_0/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth+'.branch0.1', name_tf+'/Branch_0/Conv2d_1a_3x3')
    load_conv2d(state_dict, name_pth+'.branch1.0', name_tf+'/Branch_1/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth+'.branch1.1', name_tf+'/Branch_1/Conv2d_0b_1x7')
    load_conv2d(state_dict, name_pth+'.branch1.2', name_tf+'/Branch_1/Conv2d_0c_7x1')
    load_conv2d(state_dict, name_pth+'.branch1.3', name_tf+'/Branch_1/Conv2d_1a_3x3')

def load_mixed_5(state_dict, name_pth, name_tf):
    load_conv2d(state_dict, name_pth+'.branch0', name_tf+'/Branch_0/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth+'.branch1.0', name_tf+'/Branch_1/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth+'.branch1.1', name_tf+'/Branch_1/Conv2d_0b_3x3')
    load_conv2d(state_dict, name_pth+'.branch2.0', name_tf+'/Branch_2/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth+'.branch2.1', name_tf+'/Branch_2/Conv2d_0b_3x3')
    load_conv2d(state_dict, name_pth+'.branch2.2', name_tf+'/Branch_2/Conv2d_0c_3x3')
    load_conv2d(state_dict, name_pth+'.branch3.1', name_tf+'/Branch_3/Conv2d_0b_1x1')

def load_mixed_6(state_dict, name_pth, name_tf):
    load_conv2d(state_dict, name_pth+'.branch0', name_tf+'/Branch_0/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth+'.branch1.0', name_tf+'/Branch_1/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth+'.branch1.1', name_tf+'/Branch_1/Conv2d_0b_1x7')
    load_conv2d(state_dict, name_pth+'.branch1.2', name_tf+'/Branch_1/Conv2d_0c_7x1')
    load_conv2d(state_dict, name_pth+'.branch2.0', name_tf+'/Branch_2/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth+'.branch2.1', name_tf+'/Branch_2/Conv2d_0b_7x1')
    load_conv2d(state_dict, name_pth+'.branch2.2', name_tf+'/Branch_2/Conv2d_0c_1x7')
    load_conv2d(state_dict, name_pth+'.branch2.3', name_tf+'/Branch_2/Conv2d_0d_7x1')
    load_conv2d(state_dict, name_pth+'.branch2.4', name_tf+'/Branch_2/Conv2d_0e_1x7')
    load_conv2d(state_dict, name_pth+'.branch3.1', name_tf+'/Branch_3/Conv2d_0b_1x1')

def load_mixed_7(state_dict, name_pth, name_tf):
    load_conv2d(state_dict, name_pth+'.branch0', name_tf+'/Branch_0/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth+'.branch1_0', name_tf+'/Branch_1/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth+'.branch1_1a', name_tf+'/Branch_1/Conv2d_0b_1x3')
    load_conv2d(state_dict, name_pth+'.branch1_1b', name_tf+'/Branch_1/Conv2d_0c_3x1')
    load_conv2d(state_dict, name_pth+'.branch2_0', name_tf+'/Branch_2/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth+'.branch2_1', name_tf+'/Branch_2/Conv2d_0b_3x1')
    load_conv2d(state_dict, name_pth+'.branch2_2', name_tf+'/Branch_2/Conv2d_0c_1x3')
    load_conv2d(state_dict, name_pth+'.branch2_3a', name_tf+'/Branch_2/Conv2d_0d_1x3')
    load_conv2d(state_dict, name_pth+'.branch2_3b', name_tf+'/Branch_2/Conv2d_0e_3x1')
    load_conv2d(state_dict, name_pth+'.branch3.1', name_tf+'/Branch_3/Conv2d_0b_1x1')


def load():
    state_dict={}
    
    load_conv2d(state_dict, name_pth='features.0', name_tf='Conv2d_1a_3x3')
    load_conv2d(state_dict, name_pth='features.1', name_tf='Conv2d_2a_3x3')
    load_conv2d(state_dict, name_pth='features.2', name_tf='Conv2d_2b_3x3')
    
    load_conv2d(state_dict, name_pth='features.3.conv', name_tf='Mixed_3a/Branch_1/Conv2d_0a_3x3')

    load_mixed_4a_7a(state_dict, name_pth='features.4', name_tf='Mixed_4a')

    load_conv2d(state_dict, name_pth='features.5.conv', name_tf='Mixed_5a/Branch_0/Conv2d_1a_3x3')

    load_mixed_5(state_dict, name_pth='features.6', name_tf='Mixed_5b')
    load_mixed_5(state_dict, name_pth='features.7', name_tf='Mixed_5c')
    load_mixed_5(state_dict, name_pth='features.8', name_tf='Mixed_5d')
    load_mixed_5(state_dict, name_pth='features.9', name_tf='Mixed_5e')

    load_conv2d(state_dict, name_pth='features.10.branch0', name_tf='Mixed_6a/Branch_0/Conv2d_1a_3x3')
    load_conv2d(state_dict, name_pth='features.10.branch1.0', name_tf='Mixed_6a/Branch_1/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth='features.10.branch1.1', name_tf='Mixed_6a/Branch_1/Conv2d_0b_3x3')
    load_conv2d(state_dict, name_pth='features.10.branch1.2', name_tf='Mixed_6a/Branch_1/Conv2d_1a_3x3')

    load_mixed_6(state_dict, name_pth='features.11', name_tf='Mixed_6b')
    load_mixed_6(state_dict, name_pth='features.12', name_tf='Mixed_6c')
    load_mixed_6(state_dict, name_pth='features.13', name_tf='Mixed_6d')
    load_mixed_6(state_dict, name_pth='features.14', name_tf='Mixed_6e')
    load_mixed_6(state_dict, name_pth='features.15', name_tf='Mixed_6f')
    load_mixed_6(state_dict, name_pth='features.16', name_tf='Mixed_6g')
    load_mixed_6(state_dict, name_pth='features.17', name_tf='Mixed_6h')

    load_mixed_4a_7a(state_dict, name_pth='features.18', name_tf='Mixed_7a')

    load_mixed_7(state_dict, name_pth='features.19', name_tf='Mixed_7b')
    load_mixed_7(state_dict, name_pth='features.20', name_tf='Mixed_7c')
    load_mixed_7(state_dict, name_pth='features.21', name_tf='Mixed_7d')

    load_linear(state_dict, name_pth='classif', name_tf='Logits')

    return state_dict

######################################################################
## Test
######################################################################

def test(model):
    model.eval()
    from scipy import misc
    img = misc.imread('lena_299.png')
    inputs = torch.zeros(1,299,299,3)
    inputs[0] = torch.from_numpy(img)
    inputs.transpose_(1,3)
    inputs.transpose_(2,3)
    # 1, 3, 299, 299
    outputs = model.forward(torch.autograd.Variable(inputs))
    h5f = h5py.File('dump/InceptionV4/Logits.h5', 'r')
    outputs_tf = torch.from_numpy(h5f['out'][()])
    h5f.close()
    outputs = torch.nn.functional.softmax(outputs)
    print(torch.dist(outputs.data, outputs_tf))
    return outputs
 
def test_conv2d(module, name):
    #global output_tf
    h5f = h5py.File('dump/InceptionV4/'+name+'.h5', 'r')
    output_tf = torch.from_numpy(h5f['relu_out'][()])
    output_tf.transpose_(1,3)
    output_tf.transpose_(2,3)
    h5f.close()
    def test_dist(self, input, output):
        print(name, torch.dist(output.data, output_tf))
    module.register_forward_hook(test_dist)

def test_mixed_4a_7a(module, name):
    test_conv2d(module.branch0[0], name+'/Branch_0/Conv2d_0a_1x1')
    test_conv2d(module.branch0[1], name+'/Branch_0/Conv2d_1a_3x3')
    test_conv2d(module.branch1[0], name+'/Branch_1/Conv2d_0a_1x1')
    test_conv2d(module.branch1[1], name+'/Branch_1/Conv2d_0b_1x7')
    test_conv2d(module.branch1[2], name+'/Branch_1/Conv2d_0c_7x1')
    test_conv2d(module.branch1[3], name+'/Branch_1/Conv2d_1a_3x3')

######################################################################
## Main
######################################################################

if __name__ == "__main__":

    import h5py

    model = InceptionV4()
    state_dict = load()
    model.load_state_dict(state_dict)

    # test_conv2d(model.features[0], 'Conv2d_1a_3x3')
    # test_conv2d(model.features[1], 'Conv2d_2a_3x3')
    # test_conv2d(model.features[2], 'Conv2d_2b_3x3')
    # test_conv2d(model.features[3].conv, 'Mixed_3a/Branch_1/Conv2d_0a_3x3')
    # test_mixed_4a_7a(model.features[4], 'Mixed_4a')
    
    os.system('mkdir -p save')
    torch.save(model, 'save/inceptionv4.pth')
    torch.save(state_dict, 'save/inceptionv4_state.pth')

    outputs = test(model)


