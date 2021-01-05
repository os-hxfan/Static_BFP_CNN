import math
import time
import sys
import logging
import numpy as np
logger = logging.getLogger(__name__)

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

writer = SummaryWriter("./tensorboard/weight_quant_comp")

def bfp_quantize(tensor, EXPONENT_WIDTH, MANTISSA_WIDTH, quant_dim):
    # Quantize the tensor along quant_dim as Block Floating Point
    # For activation with shape [batch, channel, heigh, widht]:
    #       quantized activation has shape [batch, num_channel_block, data]
    # For weight with shape []:
    #       quantized weight has shape [batch, num_filter_block, weight]
    #print(tensor)
    v_exponent = find_exponent(tensor, EXPONENT_WIDTH)
    #print(v_exponent)
    max_exponent = find_max_exponent(v_exponent, quant_dim)
    #print(max_exponent)
    quantized_tensor = to_exponent_mantissa_width(tensor, max_exponent, MANTISSA_WIDTH, quant_dim)
    #print(quantized_tensor)
    return quantized_tensor

def find_exp_fc(array, MANTISSA_WIDTH, EXPONENT_WIDTH, block_size=1, eps=0.0001, bins_factor=4):
    # Find the proper exponent value instead of max_exp by minimize the KL_divergence
    #   num_bins is used to construct the histogram/distribution
    #   eps is used to smooth the histogram/distribution
    # Assuming the input has shape [batch, channel]
    array = array.cuda()
    orig_shape = array.shape
    block_size = orig_shape[1] if (block_size>orig_shape[1]) else block_size # group is whole channel when group is -1
    number_of_blocks = math.ceil(orig_shape[1]/block_size)
    opt_exp = torch.empty((1))
    max_exp = torch.empty((1))
    if orig_shape[1] % block_size == 0:
        # Find the max_exp
        array = torch.reshape(array, (orig_shape[0], number_of_blocks, block_size)) 
        exp_array = find_exponent(array, EXPONENT_WIDTH)
        max_exp = find_max_exponent(exp_array, quant_dim=len(array.shape)-1)
        max_exp = find_max_exponent(max_exp, quant_dim=0)
        opt_exp = max_exp.clone()
        # Unsqeeze for quantization use
        us_max_exp = max_exp.unsqueeze(0)
        # Compute the histogram of original internal features
        orig_hist = []
        orig_max = []
        orig_min = []
        orig_num_bins = []
        min_kl_div = []
        for i in range(number_of_blocks):
            flat_array = torch.flatten(array[:, i, :].cpu())
            #print (flat_array)
            target_max_int = (int)(math.ceil(torch.max(flat_array, 0)[0].item()))
            target_min_int = (int)(math.floor(torch.min(flat_array, 0)[0].item()))
            #print ("max:", target_max_int, " min:", target_min_int)
            if (i==900):
                print ("orig:", flat_array)
            num_bins = 1 + (int)((target_max_int - target_min_int)/(2/(2**(MANTISSA_WIDTH-bins_factor))))
            target_hist = torch.histc(flat_array, bins=num_bins, min=target_min_int, max=target_max_int)
            # Smoth the target histogram
            target_hist = smooth_hist(target_hist, eps)
            # Nomalize the target histogram
            target_hist = target_hist/target_hist.sum()
            # Add information into list
            orig_hist.append(target_hist)
            orig_max.append(target_max_int)
            orig_min.append(target_min_int)
            orig_num_bins.append(num_bins)
            min_kl_div.append(sys.float_info.max)

        # Quantize accodingly, Here we only explore (max_exp-6) ~ max_exp 
        for i in range(7):
            quantized_array = to_exponent_mantissa_width(array, us_max_exp-i, MANTISSA_WIDTH,
                                                        quant_dim=len(array.shape)-1)
            for j in range(number_of_blocks):
                flat_qarray = torch.flatten(quantized_array[:, j, :].cpu())
                if (((torch.max(flat_qarray, 0))[0].item() < orig_min[j])):
                    continue
                if (j==900):
                    print ("quant:", flat_qarray)
                '''
                if ((i==2) and (j==26)):
                    print("max:", orig_max[j], " min:", orig_min[j])
                    print("orig:", torch.flatten(array[:, j, :].cpu()))
                    print("num zero:", (torch.flatten(array[:, j, :].cpu()) == 0).sum())
                    print("quantized:", flat_qarray)
                '''
                quantized_hist = torch.histc(flat_qarray, bins=orig_num_bins[j], 
                                            min=orig_min[j], max=orig_max[j])
                # Smoth the quantized histogram
                quantized_hist = smooth_hist(quantized_hist, eps)
                # Log-Nomalize the quantized histogram
                quantized_hist = quantized_hist/quantized_hist.sum()
                quantized_hist = torch.log(quantized_hist)
                # Calculate the KL-Divergence 
                kl_div = F.kl_div(quantized_hist, orig_hist[j])
                if (min_kl_div[j] > kl_div.item()):
                    opt_exp[j] = (max_exp[j]-i)
                    min_kl_div[j] = kl_div.item()
    else:
        raise ValueError("Channel is not divisible by group while determining the opt exponent list the FC")
    num_nequal = (max_exp != opt_exp).sum()
    logging.debug("After minimizing the KL divergence, %d / %d shared fc exponents are improved" % (num_nequal.item(), opt_exp.numel()))
    opt_exp = opt_exp.int().cpu().data.tolist()
    opt_exp = np.repeat(opt_exp, block_size)
    #print ("opt:", opt_exp)
    max_exp = max_exp.int().cpu().data.tolist()
    max_exp = np.repeat(max_exp, block_size)
    #print ("max:", max_exp)
    return opt_exp, max_exp

def find_exp_KL_act(array, MANTISSA_WIDTH, EXPONENT_WIDTH, group=1, eps=0.0001, bins_factor=3):
    # Find the proper exponent value instead of max_exp by minimize the KL_divergence
    #   num_bins is used to construct the histogram/distribution
    #   eps is used to smooth the histogram/distribution
    # Assuming the input has shape [batch, channel, height, width]

    # Reshape to [batch, channel, height*width]
    array = array.cuda()
    orig_shape = array.shape
    group = orig_shape[1] if (group>orig_shape[1]) else group # group is whole channel when group is -1
    number_of_blocks = math.ceil(orig_shape[1]/group)
    opt_exp = torch.empty((1))
    max_exp = torch.empty((1))
    if orig_shape[1] % group == 0:
        # Find the max_exp
        array = torch.reshape(array, (orig_shape[0], number_of_blocks, group*orig_shape[2])) 
        exp_array = find_exponent(array, EXPONENT_WIDTH)
        max_exp = find_max_exponent(exp_array, quant_dim=len(array.shape)-1)
        max_exp = find_max_exponent(max_exp, quant_dim=0)
        opt_exp = max_exp.clone()
        # Unsqeeze for quantization use
        us_max_exp = max_exp.unsqueeze(0)
        # Compute the histogram of original internal features
        orig_hist = []
        orig_max = []
        orig_min = []
        orig_num_bins = []
        min_kl_div = []
        #print("bins factor:", bins_factor)
        for i in range(number_of_blocks):
            flat_array = torch.flatten(array[:, i, :].cpu())
            float_max = torch.max(flat_array, 0)[0].item()
            float_min = torch.min(flat_array, 0)[0].item()
            #print ("orignal max", float_max, "original min", float_min)
            target_max_int = (int)(math.ceil(float_max))
            target_min_int = (int)(math.floor(float_min))
            # For mobilenet only
            target_diff = target_max_int - target_min_int
            if (target_diff < 6):
                interval = 0.02
                #num_bins = 1 + (int)((target_max_int - target_min_int)/interval) #8-2 for resnet, 8-5 for inceptionv4 8-5 for vgg16 
            else:
                #interval = (float_max - float_min)/ 128
                interval = 0.035
                #num_bins = 1 + (int)((target_max_int - target_min_int)/(2/(2**(MANTISSA_WIDTH-bins_factor)))) #8-2 for resnet, 8-5 for inceptionv4 8-5 for vgg16 
            #num_bins = 1 + (int)((target_max_int - target_min_int)/interval) #8-2 for resnet, 8-5 for inceptionv4 8-5 for vgg16 
            #num_bins = 1 + (int)((target_max_int - target_min_int)/(2/(2**(MANTISSA_WIDTH-bins_factor)))) #8-2 for resnet, 8-5 for inceptionv4 8-5 for vgg16 
            
            float_board = abs(float_max) if (abs(float_max) > abs(float_min)) else abs(float_min)
            #float_interval = (2*float_board)/64 # Indicate how accurate the distribution needs
            
            #float_interval = (2*float_board)/64 if (float_min<0) else (float_board)/64

            float_interval = (float_max-float_min)/128 # Indicate how accurate the distribution needs 70.62-128
            num_bins = 1 + (int)((target_max_int - target_min_int)/float_interval)
            
            #num_bins = 1 + (int)((target_max_int - target_min_int)/(2/(2**(MANTISSA_WIDTH-bins_factor)))) #8-2 for resnet, 8-5 for inceptionv4 8-5 for vgg16 
            target_hist = torch.histc(flat_array, bins=num_bins, min=target_min_int, max=target_max_int)
            #print ("flat array", flat_array.shape)
            # Smoth the target histogram
            target_hist = smooth_hist(target_hist, eps)
            # Nomalize the target histogram
            target_hist = target_hist/target_hist.sum()
            # Add information into list
            orig_hist.append(target_hist)
            orig_max.append(target_max_int)
            orig_min.append(target_min_int)
            orig_num_bins.append(num_bins)
            min_kl_div.append(sys.float_info.max)

        # Quantize accodingly, Here we only explore (max_exp-6) ~ max_exp 
        for i in range(3):
            quantized_array = to_exponent_mantissa_width(array, us_max_exp-i, MANTISSA_WIDTH,
                                                        quant_dim=len(array.shape)-1)
            for j in range(number_of_blocks):
                flat_qarray = torch.flatten(quantized_array[:, j, :].cpu())
                if (((torch.max(flat_qarray, 0))[0].item() < orig_min[j])):
                    continue
                quantized_hist = torch.histc(flat_qarray, bins=orig_num_bins[j], 
                                            min=orig_min[j], max=orig_max[j])
                # Smoth the quantized histogram
                quantized_hist = smooth_hist(quantized_hist, eps)
                # Log-Nomalize the quantized histogram
                quantized_hist = quantized_hist/quantized_hist.sum()
                quantized_hist = torch.log(quantized_hist)
                # Calculate the KL-Divergence 
                kl_div = F.kl_div(quantized_hist, orig_hist[j])
                if (min_kl_div[j] > kl_div.item()):
                    opt_exp[j] = (max_exp[j]-i)
                    min_kl_div[j] = kl_div.item()
    else:
        raise ValueError("Channel is not divisible by group  while determining the opt exponent list the separated activation")
    num_nequal = (max_exp != opt_exp).sum()
    logging.debug("After minimizing the KL divergence, %d / %d shared act exponents are improved" % (num_nequal.item(), opt_exp.numel()))
    opt_exp = opt_exp.int().cpu().data.tolist()
    opt_exp = np.repeat(opt_exp, group)
    max_exp = max_exp.int().cpu().data.tolist()
    max_exp = np.repeat(max_exp, group)
    #print ("kl div:", min_kl_div)
    return opt_exp, max_exp

def find_exp_act_3d(array, MANTISSA_WIDTH, EXPONENT_WIDTH, group = 1, eps=0.0001, bins_factor=3):
    # Find the proper exponent value instead of max_exp by minimize the KL_divergence
    #   num_bins is used to construct the histogram/distribution
    #   eps is used to smooth the histogram/distribution
    # Assuming the input has shape [batch, channel, height*width]
    array = array.cuda()
    orig_shape = array.shape
    group = orig_shape[1] if ((group==-1) or (group>orig_shape[1])) else group # group is whole channel when group is -1
    number_of_blocks = math.ceil(orig_shape[1]/group)
    opt_exp = []
    max_exp = []
    num_frame = orig_shape[2]
    if orig_shape[1] % group == 0:
        for i in range(num_frame):
            frame = array[:, :, i, :]
            opt_e, max_e = find_exp_KL_act(frame, MANTISSA_WIDTH, EXPONENT_WIDTH, group=group, eps=eps,
                            bins_factor=bins_factor)
            max_exp = max_exp + max_e
            opt_exp = opt_exp + opt_e
    else:
        raise NotImplementedError
    return opt_exp, max_exp

def find_exp_act(array, MANTISSA_WIDTH, EXPONENT_WIDTH, group = 1, eps=0.0001, bins_factor=3):
    # Find the proper exponent value instead of max_exp by minimize the KL_divergence
    #   num_bins is used to construct the histogram/distribution
    #   eps is used to smooth the histogram/distribution
    # Assuming the input has shape [batch, channel, height*width]
    array = array.cuda()
    orig_shape = array.shape
    group = orig_shape[1] if ((group==-1) or (group>orig_shape[1])) else group # group is whole channel when group is -1
    number_of_blocks = math.ceil(orig_shape[1]/group)
    opt_exp = []
    max_exp = []
    if orig_shape[1] % group == 0:
        opt_exp, max_exp = find_exp_KL_act(array, MANTISSA_WIDTH, EXPONENT_WIDTH, group=group, eps=eps,
                            bins_factor=bins_factor)
    else:
        logging.info('Channel is not divisible by channel group while determining the opt exponent list the activation')
        # Separate two part, tensor1 contain (number_of_blocks-1), tensor2 contain the rest
        first_chnl = ((number_of_blocks-1)*group)
        array1 = array[:, 0 : first_chnl, :]
        array2 = array[:, first_chnl : orig_shape[1], :]
        opt_exp1 = []
        max_exp1 = []
        opt_exp2 = []
        max_exp2 = []
        opt_exp1, max_exp1 = find_exp_KL_act(array1, MANTISSA_WIDTH, EXPONENT_WIDTH, group=group, eps=eps, 
                                            bins_factor=bins_factor)
        opt_exp2, max_exp2 = find_exp_KL_act(array2, MANTISSA_WIDTH, EXPONENT_WIDTH, group=group, eps=eps,
                                            bins_factor=bins_factor)
        #print ("opt1:", len(list(opt_exp1)))
        #print ("opt2:", len(opt_exp2))
        opt_exp = list(opt_exp1) + list(opt_exp2)
        max_exp = list(max_exp1) + list(max_exp2)
        # Perform quantization
    #print ("orig shape:", orig_shape[1], " exp_list_len", len(opt_exp))
    return opt_exp, max_exp

def find_bin_percent_hist(float_hist, num_bins, percentage=0.965):
    margin_min_bin = 0
    margin_max_bin = num_bins - 1
    cul_percent = 0.0
    total_num = float_hist.sum()
    cur_num = 0
    while (float_hist[margin_min_bin] == 0):
        margin_min_bin+=1
    while (float_hist[margin_max_bin] == 0):
        margin_max_bin-=1
    while (margin_min_bin != margin_max_bin):
        cur_num += float_hist[margin_min_bin]
        cur_num += float_hist[margin_max_bin]
        margin_min_bin+=1
        margin_max_bin-=1
        cur_percent = (float)(cur_num/total_num)
        #print ("cur_num:", cur_num, "total_num:", total_num)
        if ((cur_percent > (float)(1.0-0.9))):
            margin_min_bin-=1
            margin_max_bin+=1
            break
        elif ((cur_percent > (float)(1.0-percentage))):
            break
    return margin_min_bin, margin_max_bin

def find_exp_KL_weight(array, MANTISSA_WIDTH, EXPONENT_WIDTH, group=1, eps=0.0001, bins_factor=3, num_bins=64):
    # Find the proper exponent value instead of max_exp by minimize the KL_divergence
    #   num_bins is used to construct the histogram/distribution
    #   eps is used to smooth the histogram/distribution
    # Assuming the input has shape [batch, channel, height, width]

    # Reshape to [filter, channel, kernel, kernel]
    array = array.cuda()
    orig_array = array.clone()
    orig_shape = array.shape
    group = orig_shape[1] if (group>orig_shape[1]) else group # group is whole channel when group is -1
    number_of_blocks = math.ceil(orig_shape[1]/group)
    opt_exp = torch.empty((1))
    max_exp = torch.empty((1))
    #print ("weight shape", orig_shape)
    if orig_shape[1] % group == 0:
        # Find the max_exp
        array = torch.reshape(array, (orig_shape[0], number_of_blocks, group*orig_shape[2]*orig_shape[3])) 
        exp_array = find_exponent(array, EXPONENT_WIDTH)
        max_exp = find_max_exponent(exp_array, quant_dim=len(array.shape)-1)

        min_exp = find_min_exponent(exp_array, quant_dim=len(array.shape)-1)
        opt_exp = max_exp.clone()
        orig_hist = []
        orig_max = []
        orig_min = []
        orig_num_bins = []
        min_kl_div = []
        # Build distribution for each filter
        for j in range(orig_shape[0]):
            orig_hist.append([])
            orig_max.append([])
            orig_min.append([])
            orig_num_bins.append([])
            min_kl_div.append([])   
            # Build distribution for each block
            for i in range(number_of_blocks):
                if (max_exp[j][i] < 2):
                    orig_hist[j].append(None)
                    orig_max[j].append(None)
                    orig_min[j].append(None)
                    orig_num_bins[j].append(None)
                    min_kl_div[j].append(None)
                    continue
                flat_array = torch.flatten(array[j, i, :].cpu())
                float_max = torch.max(flat_array, 0)[0].item()
                float_min = torch.min(flat_array, 0)[0].item()
                #print ("orignal max", float_max, "original min", float_min)
                target_max_int = (int)(math.ceil(float_max))
                target_min_int = (int)(math.floor(float_min))
                # Build histogram with 8000 bins, get 96.5% min, max
                '''
                float_interval = 0.00001
                float_bins = 1 + (int)((target_max_int - target_min_int)/float_interval)
                float_hist = torch.histc(flat_array, bins=float_bins, min=target_min_int, max=target_max_int)
                margin_min_bin, margin_max_bin = find_bin_percent_hist(float_hist, num_bins=float_bins, percentage=0.965)
                float_min = target_min_int + margin_min_bin*float_interval
                float_max = target_max_int - (float_bins - margin_max_bin)*float_interval
                '''
                #print ("after max", float_max, "after min", float_min)
                #float_interval = (float_max-float_min)/num_bins # Indicate how accurate the distribution needs
                
                #float_interval = (target_max_int-target_min_int)/32 # Indicate how accurate the distribution needs
                #float_interval = 2**(max_exp[j][i]-6-1)
                #float_interval = 0.08
                #float_interval = 0.02 # Indicate how accurate the distribution needs
                # According to 96.5% min, max build histrogram with specified bins

                ratio = 8 if (orig_shape[1] == 1) else 8
                float_interval = (float_max-float_min)/ratio # Indicate how accurate the distribution needs  8 is best
                target_bins = 1 + (int)((target_max_int - target_min_int)/float_interval)
                '''
                bins_factor = 1
                target_bins = 1 + (int)((target_max_int - target_min_int)/(2/(2**(MANTISSA_WIDTH-bins_factor))))
                '''
                #target_bins = 1 + (int)((float_max - float_min)/float_interval)
                #print ("flat array", flat_array.shape)
                target_hist = torch.histc(flat_array, bins=target_bins, min=target_min_int, max=target_max_int)
                # Smoth the target histogram
                target_hist = smooth_hist(target_hist, eps)
                # Nomalize the target histogram
                target_hist = target_hist/target_hist.sum()
                # Add information into list
                orig_hist[j].append(target_hist)
                orig_max[j].append(target_max_int)
                orig_min[j].append(target_min_int)
                orig_num_bins[j].append(target_bins)
                min_kl_div[j].append(sys.float_info.max)

        # Usu Max to BFP quantize the array
        max_quant_array = to_exponent_mantissa_width(array, max_exp, MANTISSA_WIDTH, quant_dim=len(array.shape)-1)
        opt_quant_array = max_quant_array.clone()
        explore_range = 3 if (orig_shape[1] == 1) else 2
        #explore_range = 4 if (orig_shape[1] == 1) else 2
        # Quantize accodingly, Here we only explore (max_exp-6) ~ max_exp 
        for i in range(explore_range):
            quant_array = to_exponent_mantissa_width(array, max_exp-i, MANTISSA_WIDTH,
                                                        quant_dim=len(array.shape)-1)
            for k in range(orig_shape[0]):
                for j in range(number_of_blocks):
                    if (max_exp[k][j] < 2):
                        continue
                    unflatten_shape = quant_array[k, j, :].shape
                    flat_qarray = torch.flatten(quant_array[k, j, :].cpu())
                    if (((torch.max(flat_qarray, 0))[0].item() < orig_min[k][j]) or (((torch.min(flat_qarray, 0))[0].item() > orig_max[k][j]))):
                        continue
                    quantized_hist = torch.histc(flat_qarray, bins=orig_num_bins[k][j], 
                                                min=orig_min[k][j], max=orig_max[k][j])
                    # Smoth the quantized histogram
                    quantized_hist = smooth_hist(quantized_hist, eps)
                    # Log-Nomalize the quantized histogram
                    quantized_hist = quantized_hist/quantized_hist.sum()
                    quantized_hist = torch.log(quantized_hist)
                    # Calculate the KL-Divergence 
                    kl_div = F.kl_div(quantized_hist, orig_hist[k][j])
                    if (min_kl_div[k][j] > kl_div.item()):
                        opt_exp[k][j] = (max_exp[k][j]-i)
                        min_kl_div[k][j] = kl_div.item()
                        opt_quant_array[k, j, :] = torch.reshape(flat_qarray, unflatten_shape)
    else:
        raise ValueError("Channel is not divisible by group  while determining the opt exponent list the separated weight")
    num_nequal = (max_exp != opt_exp).sum()
    #print ("changes:", max_exp-opt_exp)
    logging.debug("After minimizing the KL divergence, %d / %d shared weight exponents are improved" % (num_nequal.item(), opt_exp.numel()))
    opt_exp = opt_exp.int().cpu().data.tolist()
    opt_exp = np.repeat(opt_exp, group, axis=1)
    max_exp = max_exp.int().cpu().data.tolist()
    max_exp = np.repeat(max_exp, group, axis=1)
    opt_quant_array = torch.reshape(opt_quant_array, orig_shape)
    max_quant_array = torch.reshape(max_quant_array, orig_shape)
    '''
    if (orig_shape[1] == 1):
        return max_quant_array 
    '''
    return opt_quant_array, opt_exp

def find_exp_weight(array, MANTISSA_WIDTH, EXPONENT_WIDTH, group = 1, eps=0.0001, num_bins=64):
    # Find the proper exponent value instead of max_exp by minimize the KL_divergence
    #   num_bins is used to construct the histogram/distribution
    #   eps is used to smooth the histogram/distribution
    # Assuming the input has shape [batch, channel, height*width]
    array = array.cuda()
    orig_shape = array.shape
    group = orig_shape[1] if ((group==-1) or (group>orig_shape[1])) else group # group is whole channel when group is -1
    number_of_blocks = math.ceil(orig_shape[1]/group)
    if orig_shape[1] % group == 0:
        quant_array, opt_exp = find_exp_KL_weight(array, MANTISSA_WIDTH, EXPONENT_WIDTH, group=group, eps=eps,
                            num_bins=num_bins)
    else:
        logging.info('Channel is not divisible by channel group while determining the opt exponent list the activation')
        # Separate two part, tensor1 contain (number_of_blocks-1), tensor2 contain the rest
        first_chnl = ((number_of_blocks-1)*group)
        array1 = array[:, 0 : first_chnl, :, :]
        array2 = array[:, first_chnl : orig_shape[1], :, :]
        quant_array1, opt_exp1 = find_exp_KL_weight(array1, MANTISSA_WIDTH, EXPONENT_WIDTH, group=group, eps=eps, 
                                            num_bins=num_bins)
        quant_array2, opt_exp2 = find_exp_KL_weight(array2, MANTISSA_WIDTH, EXPONENT_WIDTH, group=group, eps=eps,
                                            num_bins=num_bins)
        quant_array=torch.cat((quant_array1, quant_array2), 1)
        opt_exp = np.concatenate((opt_exp1, opt_exp2), axis=1)
    return quant_array, opt_exp

def find_exp_weight_3d(array, MANTISSA_WIDTH, EXPONENT_WIDTH, group = 1, eps=0.0001, num_bins=64):
    # Find the proper exponent value instead of max_exp by minimize the KL_divergence
    #   num_bins is used to construct the histogram/distribution
    #   eps is used to smooth the histogram/distribution
    # Assuming the input has shape [batch, channel, height*width]
    array = array.cuda()
    orig_shape = array.shape
    array = torch.reshape(array, (orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]*orig_shape[4])) 
    group = orig_shape[1] if ((group==-1) or (group>orig_shape[1])) else group # group is whole channel when group is -1
    number_of_blocks = math.ceil(orig_shape[1]/group)
    if orig_shape[1] % group == 0:
        quant_array, opt_exp = find_exp_KL_weight(array, MANTISSA_WIDTH, EXPONENT_WIDTH, group=group, eps=eps,
                            num_bins=num_bins)
    else:
        NotImplemented
    quant_array = torch.reshape(quant_array, (orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], orig_shape[4]))
    return quant_array, opt_exp

def bfp_quant_weight_KL(array, MANTISSA_WIDTH, EXPONENT_WIDTH, group = 1, eps=0.0001):
    # Find the proper exponent value instead of max_exp by minimize the KL_divergence
    #   num_bins is used to construct the histogram/distribution
    #   eps is used to smooth the histogram/distribution
    # Assuming the input has shape [filter, channel, k, k]
    array = array.cuda()
    orig_shape = array.shape
    group = orig_shape[1] if (group==-1) else group # group is whole channel when group is -1
    # Reshape [filter, channel, k * k]
    number_of_blocks = math.ceil(orig_shape[1]/group)
    if orig_shape[1] % group == 0:
        array = torch.reshape(array, (orig_shape[0], number_of_blocks, group*orig_shape[2]*orig_shape[3])) 
        # Find the max_exp [filter, channel]
        exp_array = find_exponent(array, EXPONENT_WIDTH)
        max_exp = find_max_exponent(exp_array, quant_dim=len(array.shape)-1)
        quantized_array = to_exponent_mantissa_width(array, max_exp, MANTISSA_WIDTH,
                                                        quant_dim=len(array.shape)-1)
        quantized_array = torch.reshape(quantized_array, (orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]))
        max_exp = max_exp.int().cpu().data.tolist()
        max_exp = np.repeat(max_exp, group, axis=1)
    else:
        logging.info('Channel is not divisible by channel group while bfp quantizeing the weight')
        if number_of_blocks == 1:
            # This means that the depth is less than the block size, so just one tensor will be created
            array = torch.reshape(array, (orig_shape[0], 1, orig_shape[1]*orig_shape[2]*orig_shape[3]))
            exp_array = find_exponent(array, EXPONENT_WIDTH)
            max_exp = find_max_exponent(exp_array, quant_dim=len(array.shape)-1)
            quantized_array = to_exponent_mantissa_width(array, max_exp, MANTISSA_WIDTH,
                                                        quant_dim=len(array.shape)-1)
            quantized_array = torch.reshape(quantized_array, (orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3]))
            max_exp = max_exp.int().cpu().data.tolist()
            max_exp = np.repeat(max_exp, orig_shape[1], axis=1)
        else:
            # Separate two part, tensor1 contain (number_of_blocks-1), tensor2 contain the rest
            first_chnl = ((number_of_blocks-1)*group)
            array1 = array[:, 0 : first_chnl, :, :]
            a1_shp = array1.shape
            array2 = array[:, first_chnl : orig_shape[1], :, :]
            a2_shp = array2.shape

            # Perform quantization
            array1 = torch.reshape(array1, (orig_shape[0], number_of_blocks-1, group*orig_shape[2]*orig_shape[3]))
            exp_array1 = find_exponent(array1, EXPONENT_WIDTH)
            max_exp1 = find_max_exponent(exp_array1, quant_dim=len(array1.shape)-1)
            quantized_array1 = to_exponent_mantissa_width(array1, max_exp1, MANTISSA_WIDTH,
                                                        quant_dim=len(array1.shape)-1)

            array2 = torch.reshape(array2, (orig_shape[0], 1, (orig_shape[1]-first_chnl)*orig_shape[2]*orig_shape[3]))
            exp_array2 = find_exponent(array2, EXPONENT_WIDTH)
            max_exp2 = find_max_exponent(exp_array2, quant_dim=len(array2.shape)-1)
            quantized_array2 = to_exponent_mantissa_width(array2, max_exp2, MANTISSA_WIDTH,
                                                        quant_dim=len(array2.shape)-1)

            # Reshape and put back to original tensor
            array1 = torch.reshape(array1, a1_shp)
            array2 = torch.reshape(array2, a2_shp)
            array[:, 0 : first_chnl, :, :] = array1 
            array[:, first_chnl : orig_shape[1], :, :] = array2
            max_exp1 = np.repeat(max_exp1, group, axis=1)
            max_exp1 = max_exp1.int().cpu().data.tolist()
            max_exp2 = np.repeat(max_exp2, group, axis=1)
            max_exp2 = max_exp2.int().cpu().data.tolist()
            max_exp = torch.cat((max_exp1, max_exp2), 1)
            quantized_array = array
    return quantized_array, max_exp

def bfp_quant_bias_KL(array, MANTISSA_WIDTH):
    # Find the proper exponent value instead of max_exp by minimize the KL_divergence
    #   eps is used to smooth the histogram/distribution
    # Assuming the input has shape [filter]
    #bias_mantissa = 2 * (MANTISSA_WIDTH-2) # The precision of mantissa is decided during the multiplication
    quantized_array = rounding_mantissa(array, MANTISSA_WIDTH)
    #print ("error:", (torch.abs(array-quantized_array)).sum())
    return quantized_array

def rounding_mantissa(array, MANTISSA_WIDTH):
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    # hardware implementation if exponent is not enough, multiply the matissa since the bit-width is 32
    assert (MANTISSA_WIDTH <= 24), "The mantissa of bias should be less equal than 24"
    shp = array.shape
    #MANTISSA_WIDTH = 23
    exponent_needed = (MANTISSA_WIDTH)*torch.ones(shp).cuda()
    first_mant_w = torch.pow(2, exponent_needed)
    array = array*first_mant_w
    # Half LSB rounding:
    array = torch.round(array)
    array = array/first_mant_w

    # Apply clamp
    #max_clamp = ((1-(1/2)**(MANTISSA_WIDTH-2))/(1-(1/2))) * torch.pow(2, maxexp)
    #max_clamp = max_clamp * torch.ones(shp).cuda()
    #array = torch.min(array, max_clamp)
    return array


def smooth_hist(array, eps=0.0001):
    # This implementation is refer to the mxnet quantization document:
    # https://github.com/apache/incubator-mxnet/blob/e17b7e2947b3848ee1b41f8ec8abafe0d1c319ad/python/mxnet/contrib/quantization.py#L241
    #print ("before smooth", array)
    is_zeros = (array == 0).float()
    is_nonzeros = (array != 0).float()
    n_zeros = is_zeros.sum()
    n_nonzeros = array.numel() - n_zeros
    if (n_nonzeros.item() == 0):
        raise ValueError("All the values are zeros, the array shape is:", array)
    eps1 = eps * n_zeros.float().item() / n_nonzeros.float().item()
    #print("eps1:", eps1)
    array = array.float()
    array += eps * is_zeros + (-eps1) * is_nonzeros
    assert (array <= 0).sum() == 0, "Some negtive values are generated during smoothing the histogram"
    return array

def find_exponent(array, EXPONENT_WIDTH):
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    MAX = 2**(EXPONENT_WIDTH-1)-1
    MIN = -2**(EXPONENT_WIDTH-1)
    absolute = torch.abs(array)
    value_log = torch.log2(absolute)
    value_log = torch.clamp(value_log, MIN, MAX)
    v_exponent = torch.floor(value_log)
    return v_exponent

def find_max_exponent(array, quant_dim):
    # Find the max exponent along the dim dimension
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    max_exponent, _ = torch.max(array, quant_dim)

    # The return is of shape [number_of_blocks, channel, h, w]
    return max_exponent

def find_min_exponent(array, quant_dim):
    # Find the max exponent along the dim dimension
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    max_exponent, _ = torch.min(array, quant_dim)

    # The return is of shape [number_of_blocks, channel, h, w]
    return max_exponent

def to_exponent_mantissa_width(array, maxexp, MANTISSA_WIDTH, quant_dim):
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    shp = array.shape
    maxexp = maxexp.unsqueeze(quant_dim)
    # NOTE THAT THIS -2 IS BECAUSE OF THE LEADING 1 AND THE FACT THAT THIS SHOULD BE IN 2s COMPLEMENT
    # Make the exponent_needed has the same shape with array
    exponent_needed = (MANTISSA_WIDTH-maxexp-2)*torch.ones(shp).cuda()
    #print (exponent_needed)
    first_mant_w = torch.pow(2, exponent_needed)
    array = array*first_mant_w
    #print (array)
    # Half LSB rounding:
    array = torch.round(array)
    # print(array[0, :, 0, 0]) # Uncomment to print integer values
    array = array/first_mant_w

    # Apply clamp
    max_clamp = ((1-(1/2)**(MANTISSA_WIDTH-2))/(1-(1/2))) * torch.pow(2, maxexp)
    max_clamp = max_clamp * torch.ones(shp).cuda()
    #print ("clamped:", (array > max_clamp).sum(), "shape:", array.shape)
    array = torch.min(array, max_clamp)

    min_clamp = -max_clamp
    array = torch.max(array, min_clamp)

    return array

