import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
# Others
import math
import numpy as np

# Internal
from models import golden_mobilenetv2
from lib import BFPActivation
from lib.BFPConvertor import BFPConvertor
from lib import BFPFullyConnet

# Pytorch
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

###V2<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def block_mobilenet(pretrained=False, num_classes=1000, bfp=False, 
        group=1, mantisa_bit=8, exp_bit=8, opt_exp_act_list=None):
    weight_exp_list = []
    golden_model = golden_mobilenetv2.MobileNetV2()
    #print("golden model:", golden_model)
    if (pretrained):
        golden_model = torch.nn.DataParallel(golden_model).cuda()
        golden_model.load_state_dict(torch.load('models/mobilenetv2_Top1_71.806_Top2_90.410.pth.tar')) 
    if (bfp):
        block_model = block_MobileNetV2(num_classes=num_classes, exp_bit=exp_bit, mantisa_bit=mantisa_bit,
                        opt_exp_act_list=opt_exp_act_list)
        block_model = torch.nn.DataParallel(block_model).cuda()
        # BFP converter
        mobilev2_converter = BFPConvertor(mantisa_bit, exp_bit)
        block_model, weight_exp_list = mobilev2_converter(golden_model, block_model, group, is_kl=True)  
        model = block_model 
    else:
        model = golden_model

    return model, weight_exp_list

class BFP_BN2D(nn.BatchNorm2d):
    def __init__(self, out_planes, padding=0, 
                exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(BFP_BN2D, self).__init__(out_planes)
        #super(BFP_BN2D, self).__init__()

        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind

    def forward(self, x):
        #x = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias)
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[self.start_exp_ind])
        return x
def block_conv_bn_relu(inp, oup, kernel_size=3, stride=1, groups=1,
                    exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, groups=groups, bias=True),
        BFP_BN2D(oup, exp_bit=exp_bit, mantisa_bit=mantisa_bit, start_exp_ind=start_exp_ind,
                opt_exp_act_list=opt_exp_act_list),
        #nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def block_conv_1x1_bn(inp, oup, exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
        BFP_BN2D(oup, exp_bit=exp_bit, mantisa_bit=mantisa_bit, start_exp_ind=start_exp_ind,
                opt_exp_act_list=opt_exp_act_list),
        #nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class block_InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,
                exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(block_InvertedResidual, self).__init__()
        self.stride = stride
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=True),
            BFP_BN2D(inp * expand_ratio, exp_bit=exp_bit, mantisa_bit=mantisa_bit, start_exp_ind=start_exp_ind,
                opt_exp_act_list=opt_exp_act_list),
            #nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=True),
            BFP_BN2D(inp * expand_ratio, exp_bit=exp_bit, mantisa_bit=mantisa_bit, start_exp_ind=start_exp_ind+1,
                opt_exp_act_list=opt_exp_act_list),
            #nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=True),
            #nn.BatchNorm2d(oup),
            BFP_BN2D(oup, exp_bit=exp_bit, mantisa_bit=mantisa_bit, start_exp_ind=start_exp_ind+2,
                opt_exp_act_list=opt_exp_act_list),
        )

    def forward(self, x):
        if self.use_res_connect:
            '''
            max_exp_act_list =  np.maximum.reduce([self.opt_exp_act_list[self.start_exp_ind+2], self.opt_exp_act_list[self.start_exp_ind-1]]).tolist()
            out = self.conv(x)
            x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit, max_exp_act_list)
            out = BFPActivation.transform_activation_offline(out, self.exp_bit, self.mantisa_bit, max_exp_act_list)
            return x + out
            '''
            out = self.conv(x)
            x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[self.start_exp_ind+2])
            return x + out
            
        else:
            return self.conv(x)


class block_MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.0,
            exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(block_MobileNetV2, self).__init__()
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind + 1 ## image input needs +1

        block = block_InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [block_conv_bn_relu(3, input_channel, stride=2, exp_bit=self.exp_bit,
                    mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind)]
        # building inverted residual blocks
        self.start_exp_ind += 1
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block_InvertedResidual(input_channel, output_channel,  s, t, exp_bit=self.exp_bit,
                        mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind))
                else:
                    self.features.append(block_InvertedResidual(input_channel, output_channel, 1, t, exp_bit=self.exp_bit,
                        mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind))
                input_channel = output_channel
                self.start_exp_ind += 3
        # building last several layers
        self.features.append(block_conv_1x1_bn(input_channel, self.last_channel, exp_bit=self.exp_bit,
                mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind))
        self.features.append(nn.AvgPool2d(int(input_size/32)))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, num_classes),
        )

    def forward(self, x): 
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[0])
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        x = BFPFullyConnet.transform_fc_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[-1])
        return x

