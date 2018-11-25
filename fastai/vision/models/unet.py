from ...torch_core import *
from ...layers import *
from ...callbacks.hooks import *

__all__ = ['DynamicUnet', 'UnetBlock']

def _get_sfs_idxs(sizes:Sizes) -> List[int]:
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    if feature_szs[0] != feature_szs[1]: sfs_idxs = [0] + sfs_idxs
    return sfs_idxs

class UnetBlockOld(nn.Module):
    "A basic U-Net block."
    def __init__(self, up_in_c:int, x_in_c:int, hook:Hook):
        super().__init__()
        self.hook = hook
        ni = up_in_c
        self.upconv = conv2d(ni, ni//2, ks=1) # H, W -> 2H, 2W
        ni = ni//2 + x_in_c
        self.conv1 = conv2d(ni, ni//2)
        self.bn1 = nn.BatchNorm2d(ni)
        ni = ni//2
        self.conv2 = conv2d(ni, ni)
        self.bn  = nn.BatchNorm2d(ni)
        self.bn2 = nn.BatchNorm2d(ni)

    def forward(self, up_in:Tensor) -> Tensor:
        s = self.hook.stored
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            up_out = F.interpolate(up_in, s.shape[-2:], mode='bilinear')
        up_out = self.upconv(up_out)
        cat_x = self.bn1(F.relu(torch.cat([up_out, s], dim=1)))
        x = self.bn2(F.relu(self.conv1(cat_x)))
        x = F.relu(self.conv2(x))
        return self.bn(x)

class UnetBlock(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    def __init__(self, up_in_c:int, x_in_c:int, hook:Hook, final_div:bool=True, all_wn:bool=False, blur:bool=False):
        super().__init__()
        self.hook = hook
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur)
        self.bn1 = nn.BatchNorm2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = ni if final_div else ni//2
        self.conv1 = conv_layer(ni, nf, wn=all_wn)
        self.conv2 = conv_layer(nf, nf, wn=all_wn)

    def forward(self, up_in:Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = F.relu(torch.cat([up_out, self.bn1(s)], dim=1))
        x = self.conv1(cat_x)
        return self.conv2(x)

class DynamicUnet(nn.Sequential):
    "Create a U-Net from a given architecture."
    def __init__(self, encoder:nn.Module, n_classes:int, all_wn:bool=False, blur:bool=False):
        imsize = (256,256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(conv_layer(ni, ni*2, wn=True),
                                    conv_layer(ni*2, ni, wn=all_wn)).eval()
        x = middle_conv(x)
        layers = [encoder, nn.BatchNorm2d(ni), nn.ReLU(), middle_conv]

        for i,idx in enumerate(sfs_idxs):
            not_final = i!=len(sfs_idxs)-1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final,
                                   all_wn=all_wn, blur=(blur and not_final)).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni))
        layers.append(conv_layer(ni, n_classes, ks=1, use_activ=False, bn=False))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()

