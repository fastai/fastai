from ...torch_core import *
from ...layers import *

__all__ = ['discriminator', 'generator', 'BasicGAN', 'WasserteinLoss']

def AvgFlatten():
    return Lambda(lambda x: x.mean(0).view(1))

def discriminator(in_size, n_channels, n_features, n_extra_layers=0):
    layers = [conv_layer(n_channels, n_features, 4, 2, 1, bn=False, leaky=0.2)]
    cur_size, cur_ftrs = in_size//2, n_features
    layers.append(nn.Sequential(*[conv_layer(cur_ftrs, cur_ftrs, 3, 1, leaky=0.2) for _ in range(n_extra_layers)]))
    while cur_size > 4:
        layers.append(conv_layer(cur_ftrs, cur_ftrs*2, 4, 2, 1, leaky=0.2))
        cur_ftrs *= 2 ; cur_size //= 2
    layers += [conv2d(cur_ftrs, 1, 4, padding=0), AvgFlatten()]
    return nn.Sequential(*layers)

def generator(in_size, noise_sz, n_channels, n_features, n_extra_layers=0):
    cur_size, cur_ftrs = 4, n_features//2
    while cur_size < in_size:  cur_size *= 2; cur_ftrs *= 2
    layers = [conv_layer(noise_sz, cur_ftrs, 4, 1, transpose=True)]
    cur_size = 4
    while cur_size < in_size // 2:
        layers.append(conv_layer(cur_ftrs, cur_ftrs//2, 4, 2, 1, transpose=True))
        cur_ftrs //= 2; cur_size *= 2
    layers += [conv_layer(cur_ftrs, cur_ftrs, 3, 1, 1, transpose=True) for _ in range(n_extra_layers)]
    layers += [conv2d_trans(cur_ftrs, n_channels, 4, 2, 1, bias=False), nn.Tanh()]
    return nn.Sequential(*layers)

class BasicGAN(nn.Module):
    
    def __init__(self, in_size, noise_sz, n_channels, n_features, n_extra_layers=0):
        super().__init__()
        self.discriminator = discriminator(in_size, n_channels, n_features, n_extra_layers)
        self.generator = generator(in_size, noise_sz, n_channels, n_features, n_extra_layers)
    
    def forward(self, x, gen=False):
        return self.generator(x) if gen else self.discriminator(x)

class WasserteinLoss(nn.Module):
    
    def forward(self, real, fake): return real[0] - fake[0]