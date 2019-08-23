import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
import torchvision
from torch.nn import functional as F
import torchvision.models as models

# Internal
from models import golden_mobilenetv2
from lib import BFPActivation
from lib.BFPConvertor import BFPConvertor
from lib import BFPFullyConnet

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class block_Linear(nn.Linear):
    def __init__(self, in_channels, out_channels, exp_bit=8, mantisa_bit=8, 
                start_exp_ind=0, opt_exp_act_list=None):
        super(block_Linear, self).__init__(in_channels, out_channels)
        #super(BFP_BN2D, self).__init__()

        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        x = BFPFullyConnet.transform_fc_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[self.start_exp_ind])
        return x

class block_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, exp_bit=8, mantisa_bit=8, 
                start_exp_ind=0, opt_exp_act_list=None):
        super(block_Conv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        #super(BFP_BN2D, self).__init__()

        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[self.start_exp_ind])
        return x

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, mantisa_bit=8, exp_bit=8, opt_exp_act_list=None):
        super(VGG, self).__init__()

        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            block_Linear(512 * 7 * 7, 4096, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                            opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=-3),
            nn.ReLU(True),
            nn.Dropout(),
            block_Linear(4096, 4096, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                            opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=-2),
            nn.ReLU(True),
            nn.Dropout(),
            block_Linear(4096, num_classes, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                            opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=-1),
        )

    def forward(self, x):
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[0])
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, exp_bit=8, mantisa_bit=8, opt_exp_act_list=None):
    layers = []
    in_channels = 3
    start_exp_ind = 1 # starting from 1 because of the input tensor of the first layer
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = block_Conv2d(in_channels, v, kernel_size=3, padding=1, start_exp_ind=start_exp_ind,
                            exp_bit=exp_bit, mantisa_bit=mantisa_bit, opt_exp_act_list=opt_exp_act_list)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            start_exp_ind+=1
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def block_vgg16(pretrained=False, num_classes=1000, bfp=False, group=1, mantisa_bit=8, exp_bit=8, opt_exp_act_list=None):
    weight_exp_list = []
    if (pretrained):
        golden_model = models.vgg16(pretrained=True)
    else:
        golden_model = models.vgg16()
    if (bfp):
        block_model = VGG(make_layers(cfgs['D'], batch_norm=False, exp_bit=exp_bit, mantisa_bit=mantisa_bit,
                opt_exp_act_list=opt_exp_act_list), exp_bit=exp_bit, mantisa_bit=mantisa_bit,
                opt_exp_act_list=opt_exp_act_list) ## Here we only use vgg16 without batchnorm
        vgg_converter = BFPConvertor(mantisa_bit, exp_bit)
        block_model, weight_exp_list = vgg_converter(golden_model, block_model, group, conv_isbias=True, is_kl=True)  
        model = block_model 
    else:
        model = golden_model
    #model = torch.nn.DataParallel(model).cuda()
    return model, weight_exp_list

def vgg11(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
