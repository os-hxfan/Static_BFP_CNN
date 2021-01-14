import math
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

import torch.nn as nn
from torch.nn.modules.utils import _triple
from lib import BFPActivation
from lib import BFPFullyConnet

import torch
import torch.nn as nn
# layer_sizes = [2, 2, 2, 2] for r3d-18
# layer_sizes = [3, 4, 6, 3] for r3d-34
######### Orig Model Define #############
class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)


        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        #self.bn = nn.BatchNorm3d(out_channels)
        #self.relu = nn.ReLU()


    def forward(self, x):
        x = self.temporal_spatial_conv(x)
        #x = self.bn(self.temporal_spatial_conv(x))
        #x = self.relu(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R3DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R3DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)

        return x.view(-1, 512)


class r3d_18(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes=[2, 2, 2, 2], block_type=SpatioTemporalResBlock, pretrained=False):
        super(r3d, self).__init__()

        self.res3d = R3DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.res3d(x)
        logits = self.linear(x)

        return logits

    def __load_pretrained_weights(self):
        p_dict = torch.load("/mnt/ccnas2/bdp/hf17/TCAD_3DCNNs/R3D-18-ucf101_epoch-99.pth.tar")
        print ("Loading from pretrained models")
        self.load_state_dict(p_dict['state_dict'])
        #for name in self.state_dict():
        #    print (name)
        #s_dict = self.state_dict()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class r3d_34(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes=[3, 4, 6, 3], block_type=SpatioTemporalResBlock, pretrained=False):
        super(r3d, self).__init__()

        self.res3d = R3DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.res3d(x)
        logits = self.linear(x)

        return logits

    def __load_pretrained_weights(self):
        p_dict = torch.load("/mnt/ccnas2/bdp/hf17/TCAD_3DCNNs/R3D-34-ucf101_epoch-99.pth.tar")
        print ("Loading from pretrained models")
        self.load_state_dict(p_dict['state_dict'])
        #for name in self.state_dict():
        #    print (name)
        #s_dict = self.state_dict()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

############################################
######### BFP Model Define #############
############################################

class BFP_SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True): # bias needs to be True for BFP module
        super(BFP_SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # self.exp_bit = exp_bit
        # self.mantisa_bit = mantisa_bit
        # self.opt_exp_act_list = opt_exp_act_list
        self.temporal_spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)
        #self.bn = nn.BatchNorm3d(out_channels)
        # self.relu = nn.ReLU()



    def forward(self, x):
        #x = self.bn(self.temporal_spatial_conv(x))
        x = self.temporal_spatial_conv(x)
        # x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
        #                                                  self.opt_exp_act_list, is_3d=True)
        # x = self.relu(x)
        return x


class BFP_SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False, exp_bit=4, mantisa_bit=8, opt_exp_act_list=None):
        super(BFP_SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = BFP_SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            #self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = BFP_SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = BFP_SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        #self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = BFP_SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
                #exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list[])
        #self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.conv1(x)
        
        res = BFPActivation.transform_activation_offline(res, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[0], is_3d=True)
        
        res = self.relu1(res)
        res = self.conv2(res)
        
        res = BFPActivation.transform_activation_offline(res, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[1], is_3d=True)
        

        if self.downsample:
            x = self.downsampleconv(x)
            #x = self.downsamplebn(self.downsampleconv(x))
        
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                        self.opt_exp_act_list[1], is_3d=True)
        

        return self.outrelu(x + res)


class BFP_SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=BFP_SpatioTemporalResBlock,
                 downsample=False, exp_bit=4, mantisa_bit=8, opt_exp_act_list=None):

        super(BFP_SpatioTemporalResLayer, self).__init__()

        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample,
            exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list[0:2])

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        cur_indx = 2
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size, exp_bit=self.exp_bit,
                mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list[cur_indx:cur_indx+2])]
            cur_indx +=2

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class BFP_R3DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.

        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=BFP_SpatioTemporalResBlock, exp_bit=4, mantisa_bit=8, opt_exp_act_list=None):
        super(BFP_R3DNet, self).__init__()


        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list
        print ("******The length of exp list:", len(self.opt_exp_act_list))
        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = BFP_SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.relu1 = nn.ReLU()
        cur_indx = 2
        next_indx = cur_indx + layer_sizes[0] * 2
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = BFP_SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type,
                exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list[cur_indx:next_indx])
        cur_indx = next_indx
        next_indx += layer_sizes[1] * 2 
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = BFP_SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True,
                exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list[cur_indx:next_indx])
        cur_indx = next_indx
        next_indx += layer_sizes[2] * 2 
        self.conv4 = BFP_SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True,
                exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list[cur_indx:next_indx])
        cur_indx = next_indx
        next_indx += layer_sizes[3] * 2 
        self.conv5 = BFP_SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True,
                exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list[cur_indx:next_indx])

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[0], is_3d=True)
        
        x = self.relu1(self.conv1(x))
        
        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[1], is_3d=True)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)

        return x.view(-1, 512)


class r3d_18_bfp(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes=[2, 2, 2, 2], block_type=BFP_SpatioTemporalResBlock, pretrained=False,
            exp_bit=4, mantisa_bit=8, opt_exp_act_list=None):
        super(r3d_bfp, self).__init__()

        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list

        self.res3d = BFP_R3DNet(layer_sizes, block_type, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list)
        self.linear = nn.Linear(512, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()
        #for name in self.state_dict():
        #    print (name)

    def forward(self, x):
        x = self.res3d(x)
        logits = self.linear(x)
        
        logits = BFPFullyConnet.transform_fc_offline(logits, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[-1])
        
        return logits

    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class r3d_34_bfp(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes, layer_sizes=[3, 4, 6, 3], block_type=BFP_SpatioTemporalResBlock, pretrained=False,
            exp_bit=4, mantisa_bit=8, opt_exp_act_list=None):
        super(r3d_bfp, self).__init__()

        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list

        self.res3d = BFP_R3DNet(layer_sizes, block_type, exp_bit=self.exp_bit, mantisa_bit=self.mantisa_bit, opt_exp_act_list=self.opt_exp_act_list)
        self.linear = nn.Linear(512, num_classes)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()
        #for name in self.state_dict():
        #    print (name)

    def forward(self, x):
        x = self.res3d(x)
        logits = self.linear(x)
        
        logits = BFPFullyConnet.transform_fc_offline(logits, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[-1])
        
        return logits

    def __load_pretrained_weights(self):
        s_dict = self.state_dict()
        for name in s_dict:
            print(name)
            print(s_dict[name].size())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.res3d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the fc layer of the net.
    """
    b = [model.linear]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    with torch.no_grad():
        net = r3d_18(101, pretrained=True)
        dev = "cpu"
        if dev == "cpu":
            inputs = torch.rand(1, 3, 16, 112, 112)
            net.cpu()
            test_iter = 100
        else:
            inputs = torch.rand(1, 3, 16, 112, 112).cuda()
            net.cuda()
            test_iter = 1000
        net.eval()
        start = time.time()
        for i in range(test_iter):
            outputs = net.forward(inputs)
        end = time.time()
        avg_time = ((end-start) * 1000) / test_iter
        print(avg_time, " ms")

