import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
import numpy as np

# Internal
from lib.BFPConvertor import BFPConvertor
from lib import BFPActivation
from lib import BFPFullyConnet
from lib import Utils
# PyTorch
import torch
import torch.nn as nn
import torchvision



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation) #enable bias for fused BN

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, exp_bit=8, mantisa_bit=8,
                 start_exp_ind=0, opt_exp_act_list=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out) 
        # disble bn for fused BN
        out = BFPActivation.transform_activation_online(out, self.exp_bit, self.mantisa_bit, -1)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = BFPActivation.transform_activation_online(out, self.exp_bit, self.mantisa_bit, -1)
        if self.downsample is not None:
            residual = self.downsample(x)
            residual = BFPActivation.transform_activation_online(residual, self.exp_bit,
                                                                    self.mantisa_bit, -1)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, exp_bit=8, mantisa_bit=8,
                start_exp_ind=0, opt_exp_act_list=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2= conv3x3(planes,planes,stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 =  nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = BFPActivation.transform_activation_online(out, self.exp_bit, self.mantisa_bit, -1)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = BFPActivation.transform_activation_online(out, self.exp_bit, self.mantisa_bit, -1)
        out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)
        out = BFPActivation.transform_activation_online(out, self.exp_bit, self.mantisa_bit, -1)

        if self.downsample is not None:
            # Get a max of two list
            #max_exp_act_list =  np.maximum.reduce([self.opt_exp_act_list[self.start_exp_ind+2], self.opt_exp_act_list[self.start_exp_ind+3]]).tolist()
            residual = self.downsample(x)
            # bfp quantize both tensor for shortcut using the max exponent list
            # since they have the same exp list, no need for realignment
            # residual = BFPActivation.transform_activation_online(residual, self.exp_bit,
            #                                                         self.mantisa_bit, self.opt_exp_act_list[self.start_exp_ind+3])
            #out = BFPActivation.transform_activation_offline(out, self.exp_bit, self.mantisa_bit, max_exp_act_list)
        # else:
            # bfp quantize both tensor for shortcut using the third exponent list
            # residual = BFPActivation.transform_activation_online(residual, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[self.start_exp_ind+2])
            # Get the exponent from out
        out_exp = Utils.find_exponent(out, self.exp_bit)
        out_exp = Utils.find_max_exponent(out_exp, quant_dim=len(out.shape)-1)
        out_exp = Utils.find_max_exponent(out_exp, quant_dim=len(out.shape)-2)
        out_exp = Utils.find_max_exponent(out_exp, quant_dim=0)
        out_exp = out_exp.int().cpu().data.tolist()
        # Get the exponent from input
        in_exp = Utils.find_exponent(residual, self.exp_bit)
        in_exp = Utils.find_max_exponent(in_exp, quant_dim=len(residual.shape)-1)
        in_exp = Utils.find_max_exponent(in_exp, quant_dim=len(residual.shape)-2)
        in_exp = Utils.find_max_exponent(in_exp, quant_dim=0)
        in_exp = in_exp.int().cpu().data.tolist()
        # Quantize accordint to the max
        max_exp =  np.maximum.reduce([out_exp, in_exp]).tolist()
        residual = BFPActivation.transform_activation_offline(residual, self.exp_bit, self.mantisa_bit, max_exp)
        out = BFPActivation.transform_activation_offline(out, self.exp_bit, self.mantisa_bit, max_exp)
        out+=residual
        out = self.relu(out)

        return out

class BlockResNet(nn.Module):
    def __init__(self, block, layers,num_classes = 1000, exp_bit=8, mantisa_bit=8, opt_exp_act_list=None):
        self.inplanes = 64
        super(BlockResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], exp_bit=self.exp_bit,
                                        mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list,
                                        start_exp_ind=2)
        self.layer2 = self._make_layer(block, 128, layers[1],stride=2, exp_bit=self.exp_bit,
                                        mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, 
                                        start_exp_ind=2 + (layers[0]*3+1))
        self.layer3 = self._make_layer(block, 256, layers[2],stride=2, exp_bit=self.exp_bit,
                                        mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list,
                                        start_exp_ind=2 + (layers[0]*3+1) + (layers[1]*3+1))
        self.layer4 = self._make_layer(block, 512, layers[3],stride=2, exp_bit=self.exp_bit,
                                        mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list,
                                        start_exp_ind=2 + (layers[0]*3+1) + (layers[1]*3+1) + (layers[2]*3+1))
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        print ("fc exponent:", self.opt_exp_act_list[-1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.constant_(m.alpha, 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, exp_bit=8, mantisa_bit=8, opt_exp_act_list=None, start_exp_ind=0):
        downsample = None
        if stride!=1 or self.inplanes !=planes*block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                # Fused BN
                #nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, exp_bit=exp_bit, 
                            mantisa_bit=mantisa_bit, opt_exp_act_list=opt_exp_act_list, start_exp_ind=start_exp_ind))
        start_exp_ind = start_exp_ind + 3 + (int)(downsample != None)
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, exp_bit=exp_bit,
                                 mantisa_bit=mantisa_bit, opt_exp_act_list=opt_exp_act_list, start_exp_ind=start_exp_ind))
            start_exp_ind = start_exp_ind + 3
        return nn.Sequential(*layers)

    def forward(self, x):
        x = BFPActivation.transform_activation_online(x, self.exp_bit, self.mantisa_bit, -1)
        x = self.conv1(x)
        #x = self.bn1(x) #Fused BN
        x = BFPActivation.transform_activation_online(x, self.exp_bit, self.mantisa_bit, -1)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = BFPFullyConnet.transform_fc_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[-1])
        x = BFPFullyConnet.transform_fc_online(x, self.exp_bit, self.mantisa_bit, -1)
        return x

    # bfp indicate if insert bfp quantization during inference
def resnet101(pretrained=False, bit_nmb=8, num_classes=1000, bfp=False, mantisa_bit=8, exp_bit=8, opt_exp_act_list=None):
    """Constructs a ResNet101 model
    """
    if (bfp):
        block_model =  BlockResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, mantisa_bit=mantisa_bit,
                                    exp_bit=exp_bit, opt_exp_act_list=opt_exp_act_list)
        if pretrained==True:
            golden_model = torchvision.models.resnet101(pretrained=True)
            resnet_converter = BFPConvertor(mantisa_bit, exp_bit)
            block_model = resnet_converter(golden_model, block_model)
    else:
        if pretrained==True:
            model = torchvision.models.resnet101(pretrained=True)
        else:
            model = torchvision.models.resnet101()
        block_model = model
    return block_model

def resnet50(pretrained=False, num_classes=1000, bfp=False,
                group=1, mantisa_bit=8, exp_bit=8, opt_exp_act_list=None):
    """ Constructs a ResNet50 model
    """
    weight_exp_list = []
    if (bfp):
        #print ("Shape of exp list:", np.shape(opt_exp_act_list))
        #print (opt_exp_act_list[0])
        block_model = BlockResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, mantisa_bit=mantisa_bit,
                                    exp_bit=exp_bit, opt_exp_act_list=opt_exp_act_list)
        if pretrained==True:
            golden_model = torchvision.models.resnet50(pretrained=True)
            resnet_converter = BFPConvertor(mantisa_bit, exp_bit)
            block_model, weight_exp_list = resnet_converter(golden_model, block_model, group, is_kl=False)
    else:
        if pretrained==True:
            model = torchvision.models.resnet50(pretrained=True)
        else:
            model = torchvision.models.resnet50()
        block_model = model
    #block_model = torch.nn.DataParallel(block_model).cuda()
    return block_model, weight_exp_list

def resnet34(pretrained=False, bit_nmb=8, num_classes=1000, bfp=False, mantisa_bit=8, exp_bit=8, opt_exp_act_list=None):
    """ Constructs a ResNet34 model
    """
    if (bfp):
        print ("Shape of exp list:", np.shape(opt_exp_act_list))
        block_model = BlockResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, mantisa_bit=mantisa_bit,
                                    exp_bit=exp_bit, opt_exp_act_list=opt_exp_act_list)
        if pretrained==True:
            golden_model = torchvision.models.resnet34(pretrained=True)
            resnet_converter = BFPConvertor(mantisa_bit, exp_bit)
            block_model = resnet_converter(golden_model, block_model)
    else:
        if pretrained==True:
            model = torchvision.models.resnet34(pretrained=True)
        else:
            model = torchvision.models.resnet34()
        block_model = model
    return block_model

