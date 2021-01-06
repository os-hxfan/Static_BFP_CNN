# Internal
from lib.Utils import *

# Others
import math
import time
import logging
logger = logging.getLogger(__name__)

# Pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class BFPConvertor:

    def __init__(self, mantisa_bit, exp_bit):
        self.mantisa_bit = mantisa_bit
        self.exp_bit = exp_bit

    def collect_bn_tensor(self, model):
        bn_weight = []
        bn_bias = []
        bn_mean = []
        bn_var = []
        bn_eps = []
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm2d):
                bn_weight.append(mod.weight.data.cuda())
                bn_bias.append(mod.bias.data.cuda())
                bn_mean.append(mod.running_mean.data.cuda())
                bn_var.append(mod.running_var.data.cuda())
                bn_eps.append(mod.eps)
        return bn_weight, bn_bias, bn_mean, bn_var, bn_eps

    def collect_fc_tensor(self, model):
        fc_weight = []
        fc_bias = []
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                fc_weight.append(mod.weight.data.cuda())
                fc_bias.append(mod.bias.data.cuda())
        return fc_weight, fc_bias              

    def collect_conv_tensor(self, model, conv_isbias):
        conv_weight = []
        conv_bias = []
        for mod in model.modules():
            if isinstance(mod, nn.Conv2d):
                conv_weight.append(mod.weight.data.cuda())
                if (conv_isbias):
                    conv_bias.append(mod.bias.data.cuda())
        return conv_weight, conv_bias             
    
    def fused_bn(self, conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, bn_eps):
        if (len(conv_weight) != len(bn_weight)):
            logging.info("%d conv, %d bn, not equal"%(len(conv_weight), len(bn_weight)))
        #print ("bias:", conv_bias)
        if (len(conv_bias) != 0):
            isbias=True
        else:
            isbias=False
        for i, conv_wtensor in enumerate(conv_weight):
            logging.debug("Fusing the No.%d conv"%(i))
            mean = bn_mean[i]
            var_sqrt = torch.sqrt(bn_var[i] + bn_eps[i])
            beta = bn_weight[i]
            gamma = bn_bias[i]
            if (isbias):
                b = conv_bias[i]
            else:
                b = mean.new_zeros(mean.shape)
                conv_bias.append(b)
            #print ("beta shape:", beta.shape, " var_sqrt shape:", var_sqrt.shape, "conv shape", conv_wtensor.shape)
            conv_weight[i] = conv_wtensor * (beta / var_sqrt).reshape([conv_wtensor.shape[0], 1, 1, 1])
            b = (b - mean)/var_sqrt * beta + gamma
            conv_bias[i] = b
        return conv_weight, conv_bias


    def __call__(self, golden_model, block_model, group, conv_isbias=False, is_kl=True):
        #print("Returning pretrained model with bit length", self.nmb_bits, "and block size of", self.bs_size)
        logging.info("Transferring the knowledge of pretrained model to Block-Floating-Point model")
        start = time.time()
        i = 0
        j = 0
        k = 0
        conv_weight, conv_bias = self.collect_conv_tensor(golden_model, conv_isbias)
        bn_weight, bn_bias, bn_mean, bn_var, bn_eps = self.collect_bn_tensor(golden_model)
        fc_weight, fc_bias = self.collect_fc_tensor(golden_model)
        weight_exp_list = []
        if (len(bn_weight) != 0):
            conv_weight, conv_bias = self.fused_bn(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, bn_eps)
        #print ("len of conv_weight:", len(conv_weight))
        #print ("len of bn_weight:", len(bn_weight))
        #print ("len of fc_weight:", len(fc_weight))
        #print ("block model:", block_model)
        #print ("golden model:", golden_model)
        for gmod, bmod in zip(golden_model.modules(), block_model.modules()):
            # Conv layer
            if isinstance(bmod, nn.Conv2d):
                #bmod.weight.data = bfp_quant_weight_KL(conv_weight[k], 8, 8, group)
                if (is_kl):
                    #bmod.weight.data = conv_weight[k]
                    #bmod.bias.data = conv_bias[k]                   
                    bmod.weight.data, opt_exp_list = find_exp_weight(conv_weight[k], self.mantisa_bit, self.exp_bit, group, eps=0.000000001, num_bins=32)
                    #opt_exp_list = opt_exp_list.int().cpu().data.tolist()
                    weight_exp_list.append(opt_exp_list)
                    if (conv_isbias or (len(bn_weight) != 0)):
                        bmod.bias.data = bfp_quant_bias_KL(conv_bias[k], 16) # set mantissa as 2*(10-2)=16, assume in hardware we can 16-bit fraction      
                else:
                    bmod.weight.data, max_exp_list = bfp_quant_weight_KL(conv_weight[k], self.mantisa_bit, self.exp_bit, group)
                    #max_exp_list = max_exp_list.int().cpu().data.tolist()
                    weight_exp_list.append(max_exp_list)
                    if (conv_isbias or (len(bn_weight) != 0)):
                        bmod.bias.data = bfp_quant_bias_KL(conv_bias[k], 16) # set mantissa as 2*(10-2)=16, assume in hardware we can 16-bit fraction
                #bmod.weight.data = conv_weight[k]
                #bmod.bias.data = conv_bias[k]
                k+=1
            # FC layer          
            if isinstance(bmod, nn.Linear):
                if (is_kl):
                    #bmod.weight.data = fc_weight[j]
                    #bmod.bias.data  = fc_bias[j]                    
                    orig_shape = fc_weight[j].shape
                    fc_weight[j] = torch.reshape(fc_weight[j], (orig_shape[0], orig_shape[1], 1, 1))
                    fc_weight[j], opt_exp_list = bfp_quant_weight_KL(fc_weight[j], self.mantisa_bit, self.exp_bit, -1) #quantize the weight of fc as whole
                    #opt_exp_list = opt_exp_list.int().cpu().data.tolist()
                    weight_exp_list.append(opt_exp_list)
                    bmod.weight.data = torch.reshape(fc_weight[j], orig_shape)
                    bmod.bias.data  = bfp_quant_bias_KL(fc_bias[j], 16)
                else:
                    orig_shape = fc_weight[j].shape
                    fc_weight[j] = torch.reshape(fc_weight[j], (orig_shape[0], orig_shape[1], 1, 1))
                    fc_weight[j], max_exp_list = bfp_quant_weight_KL(fc_weight[j], self.mantisa_bit, self.exp_bit, -1) #quantize the weight of fc as whole
                    #max_exp_list = max_exp_list.int().cpu().data.tolist()
                    weight_exp_list.append(max_exp_list)
                    bmod.weight.data = torch.reshape(fc_weight[j], orig_shape)
                    bmod.bias.data  = bfp_quant_bias_KL(fc_bias[j], 16) 
                #bmod.weight.data = fc_weight[j]
                #bmod.bias.data  = fc_bias[j]
                j+=1              
        end = time.time()
        logging.info("It took %f seconds for transfer learning" % (end-start))

        return block_model, weight_exp_list


class BFPConvertor_3D:

    def __init__(self, mantisa_bit, exp_bit):
        self.mantisa_bit = mantisa_bit
        self.exp_bit = exp_bit

    def collect_bn3d_tensor(self, model):
        bn_weight = []
        bn_bias = []
        bn_mean = []
        bn_var = []
        bn_eps = []
        for mod in model.modules():
            if isinstance(mod, nn.BatchNorm3d):
                bn_weight.append(mod.weight.data.cuda())
                bn_bias.append(mod.bias.data.cuda())
                bn_mean.append(mod.running_mean.data.cuda())
                bn_var.append(mod.running_var.data.cuda())
                bn_eps.append(mod.eps)
        return bn_weight, bn_bias, bn_mean, bn_var, bn_eps

    def collect_fc_tensor(self, model):
        fc_weight = []
        fc_bias = []
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                fc_weight.append(mod.weight.data.cuda())
                fc_bias.append(mod.bias.data.cuda())
        return fc_weight, fc_bias              

    def collect_conv_tensor(self, model, conv_isbias):
        conv_weight = []
        conv_bias = []
        for mod in model.modules():
            if isinstance(mod, nn.Conv3d):
                conv_weight.append(mod.weight.data.cuda())
                if (conv_isbias):
                    conv_bias.append(mod.bias.data.cuda())
        return conv_weight, conv_bias             


    def fused_bn3d(self, conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, bn_eps):
        if (len(conv_weight) != len(bn_weight)):
            logging.info("%d conv, %d bn, not equal"%(len(conv_weight), len(bn_weight)))
        #print ("bias:", conv_bias)
        if (len(conv_bias) != 0):
            isbias=True
        else:
            isbias=False
        for i, conv_wtensor in enumerate(conv_weight):
            logging.debug("Fusing the No.%d conv"%(i))
            mean = bn_mean[i]
            var_sqrt = torch.sqrt(bn_var[i] + bn_eps[i])
            beta = bn_weight[i]
            gamma = bn_bias[i]
            if (isbias):
                b = conv_bias[i]
            else:
                b = mean.new_zeros(mean.shape)
                conv_bias.append(b)
            #print ("beta shape:", beta.shape, " var_sqrt shape:", var_sqrt.shape, "conv shape", conv_wtensor.shape)
            conv_weight[i] = conv_wtensor * (beta / var_sqrt).reshape([conv_wtensor.shape[0], 1, 1, 1, 1])
            b = (b - mean)/var_sqrt * beta + gamma
            conv_bias[i] = b
        return conv_weight, conv_bias

    def __call__(self, golden_model, block_model, group, conv_isbias=False, is_kl=True):
        #print("Returning pretrained model with bit length", self.nmb_bits, "and block size of", self.bs_size)
        logging.info("Transferring the knowledge of pretrained model to Block-Floating-Point model")
        start = time.time()
        i = 0
        j = 0
        k = 0
        conv_weight, conv_bias = self.collect_conv_tensor(golden_model, conv_isbias)
        bn_weight, bn_bias, bn_mean, bn_var, bn_eps = self.collect_bn3d_tensor(golden_model)
        fc_weight, fc_bias = self.collect_fc_tensor(golden_model)
        weight_exp_list = []
        if (len(bn_weight) != 0):
            conv_weight, conv_bias = self.fused_bn3d(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, bn_eps)
        for gmod, bmod in zip(golden_model.modules(), block_model.modules()):
            # Conv layer
            if isinstance(bmod, nn.Conv3d):
                #bmod.weight.data = bfp_quant_weight_KL(conv_weight[k], 8, 8, group)
                if (is_kl):
                    #bmod.weight.data = conv_weight[k]
                    #bmod.bias.data = conv_bias[k]                   
                    bmod.weight.data, opt_exp_list = find_exp_weight_3d(conv_weight[k], self.mantisa_bit, self.exp_bit, group, eps=0.000000001, num_bins=32)
                    #opt_exp_list = opt_exp_list.int().cpu().data.tolist()
                    weight_exp_list.append(opt_exp_list)
                    if (conv_isbias):
                        bmod.bias.data = bfp_quant_bias_KL(conv_bias[k], 16) # set mantissa as 2*(10-2)=16, assume in hardware we can 16-bit fraction      
                else:
                    bmod.weight.data, max_exp_list = bfp_quant_weight_KL(conv_weight[k], self.mantisa_bit, self.exp_bit, group)
                    #max_exp_list = max_exp_list.int().cpu().data.tolist()
                    weight_exp_list.append(max_exp_list)
                    if (conv_isbias):
                        bmod.bias.data = bfp_quant_bias_KL(conv_bias[k], 16) # set mantissa as 2*(10-2)=16, assume in hardware we can 16-bit fraction
                #bmod.weight.data = conv_weight[k]
                #bmod.bias.data = conv_bias[k]
                k+=1
            # FC layer          
            if isinstance(bmod, nn.Linear):
                if (is_kl):
                    #bmod.weight.data = fc_weight[j]
                    #bmod.bias.data  = fc_bias[j]                    
                    orig_shape = fc_weight[j].shape
                    fc_weight[j] = torch.reshape(fc_weight[j], (orig_shape[0], orig_shape[1], 1, 1))
                    fc_weight[j], opt_exp_list = bfp_quant_weight_KL(fc_weight[j], self.mantisa_bit, self.exp_bit, -1) #quantize the weight of fc as whole
                    #opt_exp_list = opt_exp_list.int().cpu().data.tolist()
                    weight_exp_list.append(opt_exp_list)
                    bmod.weight.data = torch.reshape(fc_weight[j], orig_shape)
                    bmod.bias.data  = bfp_quant_bias_KL(fc_bias[j], 16)
                else:
                    orig_shape = fc_weight[j].shape
                    fc_weight[j] = torch.reshape(fc_weight[j], (orig_shape[0], orig_shape[1], 1, 1))
                    fc_weight[j], max_exp_list = bfp_quant_weight_KL(fc_weight[j], self.mantisa_bit, self.exp_bit, -1) #quantize the weight of fc as whole
                    #max_exp_list = max_exp_list.int().cpu().data.tolist()
                    weight_exp_list.append(max_exp_list)
                    bmod.weight.data = torch.reshape(fc_weight[j], orig_shape)
                    bmod.bias.data  = bfp_quant_bias_KL(fc_bias[j], 16) 
                #bmod.weight.data = fc_weight[j]
                #bmod.bias.data  = fc_bias[j]
                j+=1              
        end = time.time()
        logging.info("It took %f seconds for transfer learning" % (end-start))

        return block_model, weight_exp_list
