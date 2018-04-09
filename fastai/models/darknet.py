import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBN(nn.Module):
    #convolutional layer then Batchnorm
    def __init__(self, ch_in, ch_out, kernel_size = 3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)
    
class DarknetBlock(nn.Module):
    #The basic blocs.   
    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in//2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x
    
class Darknet(nn.Module):
    #Replicates the table 1 from the YOLOv3 paper
    def __init__(self, num_blocks, num_classes=1000):
        super().__init__()
        self.conv = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self.make_group_layer(32, num_blocks[0])
        self.layer2 = self.make_group_layer(64, num_blocks[1], stride=2)
        self.layer3 = self.make_group_layer(128, num_blocks[2], stride=2)
        self.layer4 = self.make_group_layer(256, num_blocks[3], stride=2)
        self.layer5 = self.make_group_layer(512, num_blocks[4], stride=2)
        self.linear = nn.Linear(1024, num_classes)

    def make_group_layer(self, ch_in, num_blocks, stride=1):
        layers = [ConvBN(ch_in,ch_in*2,stride=stride)]
        for i in range(num_blocks):
            layers.append(DarknetBlock(ch_in*2))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return F.log_softmax(self.linear(out))   
    
def Darknet53(): return Darknet([1,2,8,8,4])