from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
import sys

# Internal
from lib import BFPActivation
from lib.BFPConvertor import BFPConvertor
from lib import BFPFullyConnet

__all__ = ['InceptionV4', 'inceptionv4']

pretrained_settings = {
    'inceptionv4': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class block_BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, 
                exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(block_BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind

    def forward(self, x):
        x = self.conv(x)
        #x = self.bn(x)
        #print ("Before:", x)
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[self.start_exp_ind])
        #print ("After", x)
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


class block_Mixed_3a(nn.Module):

    def __init__(self, exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(block_Mixed_3a, self).__init__()
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = block_BasicConv2d(64, 96, kernel_size=3, stride=2, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit, 
                                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind)

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


class block_Mixed_4a(nn.Module):

    def __init__(self, exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(block_Mixed_4a, self).__init__()
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind

        self.branch0 = nn.Sequential(
            block_BasicConv2d(160, 64, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit, 
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+1),
            block_BasicConv2d(64, 96, kernel_size=3, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+2)
        )

        self.branch1 = nn.Sequential(
            block_BasicConv2d(160, 64, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+3),
            block_BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3), exp_bit=self.exp_bit,
                        mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+4),
            block_BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0), exp_bit=self.exp_bit,
                        mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+5),
            block_BasicConv2d(64, 96, kernel_size=(3,3), stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+6)
        )

    def forward(self, x):
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[self.start_exp_ind])
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


class block_Mixed_5a(nn.Module):

    def __init__(self, exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(block_Mixed_5a, self).__init__()
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind       
        self.conv = block_BasicConv2d(192, 192, kernel_size=3, stride=2, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                                opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+1)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[self.start_exp_ind])
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

class block_Inception_A(nn.Module):

    def __init__(self, exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(block_Inception_A, self).__init__()
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind
        self.branch0 = block_BasicConv2d(384, 96, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                                    opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+1)

        self.branch1 = nn.Sequential(
            block_BasicConv2d(384, 64, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+2),
            block_BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+3)
        )

        self.branch2 = nn.Sequential(
            block_BasicConv2d(384, 64, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+4),
            block_BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+5),
            block_BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+6)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            block_BasicConv2d(384, 96, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+7)
        )

    def forward(self, x):
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[self.start_exp_ind])
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


class block_Reduction_A(nn.Module):

    def __init__(self, exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(block_Reduction_A, self).__init__()
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind
        self.branch0 = block_BasicConv2d(384, 384, kernel_size=3, stride=2, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit+1,
                                    opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+1)

        self.branch1 = nn.Sequential(
            block_BasicConv2d(384, 192, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+2),
            block_BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+3),
            block_BasicConv2d(224, 256, kernel_size=3, stride=2, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+4)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[self.start_exp_ind])
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

class block_Inception_B(nn.Module):

    def __init__(self, exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(block_Inception_B, self).__init__()
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind
        self.branch0 = block_BasicConv2d(1024, 384, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                                    opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+1)

        self.branch1 = nn.Sequential(
            block_BasicConv2d(1024, 192, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+2),
            block_BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3), 
                        exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+3),
            block_BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0), exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+4)
        )

        self.branch2 = nn.Sequential(
            block_BasicConv2d(1024, 192, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+5),
            block_BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0), exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+6),
            block_BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3), exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+7),
            block_BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0), exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+8),
            block_BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3), exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+9)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            block_BasicConv2d(1024, 128, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+10)
        )

    def forward(self, x):
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[self.start_exp_ind])
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

class block_Reduction_B(nn.Module):

    def __init__(self, exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(block_Reduction_B, self).__init__()
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind

        self.branch0 = nn.Sequential(
            block_BasicConv2d(1024, 192, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+1),
            block_BasicConv2d(192, 192, kernel_size=3, stride=2, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+2)
        )

        self.branch1 = nn.Sequential(
            block_BasicConv2d(1024, 256, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+3),
            block_BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3), exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+4),
            block_BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0), exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+5),
            block_BasicConv2d(320, 320, kernel_size=3, stride=2, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+6)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[self.start_exp_ind])
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

class block_Inception_C(nn.Module):

    def __init__(self, exp_bit=8, mantisa_bit=8, start_exp_ind=0, opt_exp_act_list=None):
        super(block_Inception_C, self).__init__()
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = start_exp_ind

        self.branch0 = block_BasicConv2d(1536, 256, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                                    opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+1)

        self.branch1_0 = block_BasicConv2d(1536, 384, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                                    opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+2)
        self.branch1_1a = block_BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1), exp_bit=self.exp_bit,
                                    mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+3)
        self.branch1_1b = block_BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0), exp_bit=self.exp_bit,
                                    mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+4)

        self.branch2_0 = block_BasicConv2d(1536, 384, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                                    opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+5)
        self.branch2_1 = block_BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0), exp_bit=self.exp_bit,
                                    mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+6)
        self.branch2_2 = block_BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1), exp_bit=self.exp_bit,
                                    mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+7)
        self.branch2_3a = block_BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1), exp_bit=self.exp_bit,
                                    mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+8)
        self.branch2_3b = block_BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0), exp_bit=self.exp_bit,
                                    mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+9)

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            block_BasicConv2d(1536, 256, kernel_size=1, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                        opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+10)
        )

    def forward(self, x):
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[self.start_exp_ind])
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
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
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
            Inception_C()
        )
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        #Allows image of any size to be processed
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

class block_InceptionV4(nn.Module):

    def __init__(self, num_classes=1001, exp_bit=8, mantisa_bit=8, opt_exp_act_list=None):
        super(block_InceptionV4, self).__init__()
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        self.start_exp_ind = 0
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.features = nn.Sequential(
            block_BasicConv2d(3, 32, kernel_size=3, stride=2, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                                opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+1),
            block_BasicConv2d(32, 32, kernel_size=3, stride=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                                opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+2),
            block_BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                                opt_exp_act_list=self.opt_exp_act_list, start_exp_ind=self.start_exp_ind+3),

            block_Mixed_3a(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                     start_exp_ind=self.start_exp_ind+4, opt_exp_act_list=self.opt_exp_act_list),
            block_Mixed_4a(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+5, opt_exp_act_list=self.opt_exp_act_list),
            block_Mixed_5a(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+12, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_A(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+14, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_A(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+22, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_A(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+30, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_A(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+38, opt_exp_act_list=self.opt_exp_act_list),
            block_Reduction_A(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+46, opt_exp_act_list=self.opt_exp_act_list), # Mixed_6a
            block_Inception_B(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+51, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_B(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+62, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_B(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+73, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_B(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+84, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_B(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+95, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_B(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+106, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_B(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+117, opt_exp_act_list=self.opt_exp_act_list),
            block_Reduction_B(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+128, opt_exp_act_list=self.opt_exp_act_list), # Mixed_7a
            block_Inception_C(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+135, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_C(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+146, opt_exp_act_list=self.opt_exp_act_list),
            block_Inception_C(exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit,
                    start_exp_ind=self.start_exp_ind+157, opt_exp_act_list=self.opt_exp_act_list),
        )
        ## Totally should be 139+10=149
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        #Allows image of any size to be processed
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[0])
        x = self.features(x)
        x = self.logits(x)
        x = BFPFullyConnet.transform_fc_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[-1])
        return x


def inceptionv4(num_classes=1000, pretrained=False, bfp=False, group=1, mantisa_bit=8, exp_bit=8, opt_exp_act_list=None):
    weight_exp_list = []
    if pretrained:
        pretrained = 'imagenet'
        settings = pretrained_settings['inceptionv4'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionV4(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionV4(num_classes=num_classes)  
      
    if (bfp):
        golden_model = model
        if pretrained:
            #pretrained = 'imagenet'
            settings = pretrained_settings['inceptionv4'][pretrained]
            assert num_classes == settings['num_classes'], \
                "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

            # Insert bfp quantization into the Inception-v4
            block_model = block_InceptionV4(num_classes=num_classes, exp_bit=exp_bit, mantisa_bit=mantisa_bit,
                                opt_exp_act_list=opt_exp_act_list)

            block_model.input_space = settings['input_space']
            block_model.input_size = settings['input_size']
            block_model.input_range = settings['input_range']
            block_model.mean = settings['mean']
            block_model.std = settings['std']
            # BFP converter
            inceptionv4_converter = BFPConvertor(mantisa_bit, exp_bit)
            block_model, weight_exp_list = inceptionv4_converter(golden_model, block_model, group, is_kl=True)  
            model = block_model         
    #model = torch.nn.DataParallel(model).cuda()
    return model, weight_exp_list
