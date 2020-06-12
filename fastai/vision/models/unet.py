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

def get_unet_config(model, img_size=(512, 512))->Tuple[List[Tuple], List[nn.Module]]:
    "Cut the network to several blocks, the width and high of the image are reduced by half. And the image W and H >= 7"
    x = torch.rand(1, in_channels(model), *img_size)
    hooks = []
    count = 0
    layer_mata = []
    layers = []

    def flatten_moduleList(module: nn.Module) -> List[nn.Module]:
        "If the ModuleList can be found in children, flatten it. Which is important to take efficientnet as decoder of unet"
        res_list = []
        for item in module.children():
            if isinstance(item, nn.ModuleList):
                res_list.extend(flatten_moduleList(item))
            else:
                res_list.append(item)
        return res_list

    def hook(module, input, output):
        "To get the metadata of the layer"
        nonlocal count
        if len(output.shape) == 4:
            b, c, w, h = output.shape
            layer_mata.append((count, type(module).__name__, c, w, h, output.shape))
        layers.append(module)
        count += 1

    for module in flatten_moduleList(model):
        hooks.append(module.register_forward_hook(hook))
    model(x)
    for h in hooks: h.remove()

    img_size = [x.shape[-1] // (2 ** i) for i in range(8)]
    img_size = [size for size in img_size if size >= 7]
    layer_mata = pd.DataFrame(layer_mata, columns=['sn', 'layer_name', 'c', 'w', 'h', 'size'])
    layer_mata = layer_mata.loc[(layer_mata.h.isin(img_size))].drop_duplicates(['h'], keep='last')

    layer_size = list(layer_mata['size'])
    layers = [layers[i] for i in layer_mata.sn]
    return layer_size, layers

class UnetBlock(Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."
    def __init__(self, up_in_c:int, x_in_c:int, hook:Hook, final_div:bool=True, blur:bool=False, leaky:float=None,
                 self_attention:bool=False, **kwargs):
        self.hook = hook
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, leaky=leaky, **kwargs)
        self.bn = batchnorm_2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = ni if final_div else ni//2
        self.conv1 = conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = conv_layer(nf, nf, leaky=leaky, self_attention=self_attention, **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in:Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class DynamicUnet(SequentialEx):
    "Create a U-Net from a given architecture."

    def __init__(self, encoder: nn.Module, n_classes: int, img_size: Tuple[int, int] = (256, 256),
                 blur: bool = False,
                 blur_final=True, self_attention: bool = False,
                 y_range: Optional[Tuple[float, float]] = None,
                 last_cross: bool = True, bottle: bool = False, **kwargs):
        imsize = tuple(img_size)
        sfs_szs, select_layer = get_unet_config(encoder, img_size)
        ni = sfs_szs[-1][1]
        sfs_szs = list(reversed(sfs_szs[:-1]))
        select_layer = list(reversed(select_layer[:-1]))
        self.sfs = hook_outputs(select_layer, detach=False)
        x = dummy_eval(encoder, imsize).detach()

        middle_conv = nn.Sequential(conv_layer(ni, ni * 2, **kwargs),
                                    conv_layer(ni * 2, ni, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        for i, x_size in enumerate(sfs_szs):
            not_final = i != len(sfs_szs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(x_size[1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i==len(sfs_szs) - 3)
            unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, blur=do_blur, self_attention=sa,
                                   **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]: layers.append(PixelShuffle_ICNR(ni, **kwargs))
        x = PixelShuffle_ICNR(ni)(x)
        if imsize != x.shape[-2:]: layers.append(Lambda(lambda x: F.interpolate(x, imsize, mode='nearest')))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, **kwargs))
        layers += [conv_layer(ni, n_classes, ks=1, use_activ=False, **kwargs)]
        if y_range is not None: layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()

