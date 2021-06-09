# Internal
from models import model_factory
from lib import Stat_Collector
from lib import Utils
from models import inceptionv4
#from models import bfp_modules

# The Basic Library
import argparse 
import os
import logging
import numpy as np
import sys
import copy
import math
import time
from lib import BFPActivation

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Other Required Library
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tensorboardX import SummaryWriter
import pretrainedmodels

writer = SummaryWriter("./tensorboard/statistics")

modules_map = {  "BatchNorm2d" : nn.BatchNorm2d,
                 "Linear" : nn.Linear,
                 "Conv2d" :  nn.Conv2d,
                 "Inception_C" : inceptionv4.Inception_C,
                 "Reduction_B" : inceptionv4.Reduction_B,
                 "Inception_B" : inceptionv4.Inception_B,
                 "Reduction_A" : inceptionv4.Reduction_A,
                 "Inception_A" : inceptionv4.Inception_A,
                 "Mixed_5a" : inceptionv4.Mixed_5a,
                 "Mixed_4a" : inceptionv4.Mixed_4a
}

# Perform the Block Floting Quantization(BFP) on given model
def bfp_quant(model_name, dataset_dir, num_classes, gpus, mantisa_bit, exp_bit, batch_size=1, 
                num_bins=8001, eps=0.0001, num_workers=2, num_examples=10, std=None, mean=None,
                resize=256, crop=224, exp_act=None, bfp_act_chnl=1, bfp_weight_chnl=1, bfp_quant=1,
                target_module_list=None, act_bins_factor=3, fc_bins_factor=4, is_online=0):
    # Setting up gpu environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)


    # Setting up dataload for evaluation
    valdir = os.path.join(dataset_dir, 'val')
    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    # for collect intermediate data use
    collect_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=num_examples, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    # for validate the bfp model use
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)


    # Loading the model
    model, _ = model_factory.get_network(model_name, pretrained=True)
    # Insert the hook to record the intermediate result
    #target_module_list = [nn.BatchNorm2d,nn.Linear] # Insert hook after BN and FC
    model, intern_outputs = Stat_Collector.insert_hook(model, target_module_list)
    #model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    

    # Collect the intermediate result while running number of examples
    logging.info("Collecting the statistics while running image examples....")
    images_statistc = torch.empty((1))
    with torch.no_grad():
        for i_batch, (images, lables) in enumerate(collect_loader):
            images = images.cuda()
            outputs = model(images)
            #print(lables)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu() # needs to verify if this line can be deleted  
            # Collect the input data
            image_shape = images.shape
            images_statistc = torch.reshape(images, 
                                    (image_shape[0], image_shape[1], image_shape[2]*image_shape[3]))
            break
    
    # Deternmining the optimal exponent of activation and
    # Constructing the distribution for tensorboardX visualization
    logging.info("Determining the optimal exponent by minimizing the KL divergence....")
    start = time.time()
    opt_exp_act_list = []
    max_exp_act_list = []
    # For original input
    opt_exp, max_exp = Utils.find_exp_act(images_statistc, mantisa_bit, exp_bit, group=3, eps=eps, bins_factor=act_bins_factor)
    opt_exp_act_list.append(opt_exp)
    max_exp_act_list.append(max_exp)
    sc_layer_num = [7,10,17,20,23,30,33,36,39,42,49,52]
    ds_sc_layer_num = [14,27,46]
    mobilev2_sc_layer_num = [9,15,18,24,27,30,36,39,45,48]
    for i, intern_output in enumerate(intern_outputs):
        #print ("No.", i, " ", intern_output.out_features.shape)
        #Deternmining the optimal exponent by minimizing the KL_Divergence in channel-wise manner
        if (isinstance(intern_output.m, nn.Conv2d) or isinstance(intern_output.m, nn.BatchNorm2d)):
            intern_shape = intern_output.out_features.shape
            #print (intern_shape, "No.", i)
            # assmue internal activation has shape: (batch, channel, height, width)
            
            if ((model_name=="resnet50") and (i in sc_layer_num)):
                #print ("Before:", intern_shape[1])
                intern_features1 = intern_output.out_features
                intern_features2 = intern_outputs[i-3].out_features
                intern_features = torch.cat((intern_features1, intern_features2), 0)
                intern_features = torch.reshape(intern_features, (2*intern_shape[0], intern_shape[1],
                                                intern_shape[2]*intern_shape[3]))
                #print (intern_features.shape)
                opt_exp, max_exp = Utils.find_exp_act(intern_features, mantisa_bit, exp_bit, 
                                                group = bfp_act_chnl, eps=eps, bins_factor=act_bins_factor)
                opt_exp_act_list.append(opt_exp)
                max_exp_act_list.append(max_exp)
                #print ("After:", len(opt_exp))
            elif ((model_name=="resnet50") and (i in ds_sc_layer_num)):
                intern_features1 = intern_output.out_features
                intern_features2 = intern_outputs[i-1].out_features
                intern_features = torch.cat((intern_features1, intern_features2), 0)
                intern_features = torch.reshape(intern_features, (2*intern_shape[0], intern_shape[1],
                                                intern_shape[2]*intern_shape[3]))
                #print (intern_features.shape)
                opt_exp, max_exp = Utils.find_exp_act(intern_features, mantisa_bit, exp_bit, 
                                                group = bfp_act_chnl, eps=eps, bins_factor=act_bins_factor)
                #print ("Current shape", np.shape(opt_exp), " No.", i)
                #print ("Previous shape", np.shape(opt_exp_act_list[i]), " No.", i-1)
                opt_exp_act_list.append(opt_exp)
                max_exp_act_list.append(max_exp)
                opt_exp_act_list[i]=(opt_exp) #Starting from 1
                max_exp_act_list[i]=(max_exp) 
            elif ((model_name=="mobilenetv2") and (i in mobilev2_sc_layer_num)):
                intern_features1 = intern_output.out_features
                intern_features2 = intern_outputs[i-3].out_features
                intern_features = torch.cat((intern_features1, intern_features2), 0)
                intern_features = torch.reshape(intern_features, (2*intern_shape[0], intern_shape[1],
                                                intern_shape[2]*intern_shape[3]))
                #print (intern_features.shape)
                opt_exp, max_exp = Utils.find_exp_act(intern_features, mantisa_bit, exp_bit, 
                                                group = bfp_act_chnl, eps=eps, bins_factor=act_bins_factor)
                opt_exp_act_list.append(opt_exp) ##changed
                max_exp_act_list.append(max_exp)
            else:
                intern_features = torch.reshape(intern_output.out_features, 
                                (intern_shape[0], intern_shape[1], intern_shape[2]*intern_shape[3]))
                opt_exp, max_exp = Utils.find_exp_act(intern_features, mantisa_bit, exp_bit, 
                                                group = bfp_act_chnl, eps=eps, bins_factor=act_bins_factor)
                opt_exp_act_list.append(opt_exp) ##changed
                max_exp_act_list.append(max_exp)
                # ploting the distribution
                #writer.add_histogram("layer%d" % (i), intern_output.out_features.cpu().data.numpy(), bins='auto')
                quant_tensor = BFPActivation.transform_activation_offline(intern_output.out_features, exp_bit, mantisa_bit, max_exp)
                #writer.add_histogram("layer%d" % (i), quant_tensor.cpu().data.numpy(), bins='auto')
                quant_tensor = BFPActivation.transform_activation_offline(intern_output.out_features, exp_bit, mantisa_bit, opt_exp)
                writer.add_histogram("layer%d" % (i), quant_tensor.cpu().data.numpy(), bins='auto')
                
            #print (np.shape(opt_exp), " No.", i)
        elif (isinstance(intern_output.m, nn.Linear)):
            intern_shape = intern_output.out_features.shape
            opt_exp, max_exp = Utils.find_exp_fc(intern_output.out_features, mantisa_bit, exp_bit, block_size = intern_shape[1], eps=eps, bins_factor=fc_bins_factor)
            #print ("shape of fc exponent:", np.shape(opt_exp))
            opt_exp_act_list.append(max_exp)
            max_exp_act_list.append(max_exp)
        else:
            intern_shape = intern_output.in_features[0].shape
            intern_features = torch.reshape(intern_output.in_features[0], 
                                (intern_shape[0], intern_shape[1], intern_shape[2]*intern_shape[3]))
            opt_exp, max_exp = Utils.find_exp_act(intern_features, mantisa_bit, exp_bit, 
                                                group = bfp_act_chnl, eps=eps, bins_factor=act_bins_factor)
            opt_exp_act_list.append(opt_exp)
            max_exp_act_list.append(max_exp)
            
        #logging.info("The internal shape: %s" % ((str)(intern_output.out_features.shape)))
    end = time.time()
    logging.info("It took %f second to determine the optimal shared exponent for each block." % ((float)(end-start)))
    logging.info("The shape of collect exponents: %s" % ((str)(np.shape(opt_exp_act_list))))

    # Building a BFP model by insert BFPAct and BFPWeiht based on opt_exp_act_list
    torch.cuda.empty_cache() 
    if (exp_act=='kl'):
        exp_act_list = opt_exp_act_list
    else:
        exp_act_list = max_exp_act_list
    if (is_online == 1):
        model_name = "br_" + model_name
    bfp_model, weight_exp_list = model_factory.get_network(model_name, pretrained=True, bfp=(bfp_quant==1), group=bfp_weight_chnl, mantisa_bit=mantisa_bit, 
                exp_bit=exp_bit, opt_exp_act_list=exp_act_list)
    
    writer.close()
    


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="resnet34", type=str, required=True, 
                        help="Select the model for bfp quant now only support one of {resnet34,resnet50,resnet101,inceptionv4, mobilenetv2}")
    parser.add_argument("--dataset_dir", default="/dataset/", type=str, required=True,
                        help="Dataset to evaluate the bfp_quantied model. Pls use absolute path point to the dataset")
    parser.add_argument("--mantisa_bit", default=8, required=True, type=int, help="The bitwidth of mantissa in block floating point representation")
    parser.add_argument("--exp_bit", default=8, required=True, type=int, help="The bitwidth of mantissa in block floating point representation")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size on each gpu when evaluation of bfp_quantied model")
    parser.add_argument("--num_bins", default=8001, type=int, help="Used to construct the histogram/distribution for intermidate results")
    parser.add_argument("--eps", default=0.0001, type=float, help="Used to smooth the histogram/distribution")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers in data loader when evaluation of bfp_quantied model")
    parser.add_argument("--num_examples", default=10, type=int, help="Number of examples to collect internal outputs for bfp quant")
    parser.add_argument("--num_classes", default=1000, type=int, required=True, help="The number of classes when evaluation of bfp_quantied model")
    parser.add_argument("--gpus", default="0,1", type=str, required=True, help="GPUs id, separated by comma withougt space, for example: 0,1,2")
    parser.add_argument("--std", default="0.229,0.224,0.225", type=str, help="std values for image preprocessing")
    parser.add_argument("--mean", default="0.485,0.456,0.406", type=str, help="mean values for image preprocessing")
    parser.add_argument("--resize", default=256, type=int, help="The size of resized image, resize shoule be lager than crop")
    parser.add_argument("--crop", default=224, type=int, help="The size of cropped image, crop should be less than resize")
    parser.add_argument("--exp_act", default="kl", type=str, help="The way to determine the exponents in activation, suppor {max, kl}")
    parser.add_argument("--bfp_act_chnl", default=1, type=int, help="Number of channels per block in activation, -1 means whole")
    parser.add_argument("--bfp_weight_chnl", default=1, type=int, help="Number of channels per block in weight, -1 means whole")
    parser.add_argument("--bfp_quant", default=1, type=int, help="1 indicate using bfp model, 0 indicate using floating-point model")
    parser.add_argument("--hooks", default="Conv2d,Linear", type=str, help="The name of hooked nn modules, one of{BatchNorm2d,Linear,Conv2d}")
    parser.add_argument("--act_bins_factor", default=3, type=int, help="The bins_factor for constructing act histogram")
    parser.add_argument("--fc_bins_factor", default=3, type=int, help="The bins_factor for constructing act histogram")
    parser.add_argument("--is_online", default=0, type=int, help="Use online BFP quantization for benchmark")
    

    args = parser.parse_args()

    # Split the argument
    gpus = args.gpus.split(",")
    gpus_num = len(gpus)
    logging.info("totally {} gpus are using".format(gpus_num))
    # String to float
    std_str = args.std.split(",")
    std = []
    for std_v in std_str:
        std.append((float)(std_v))

    mean_str = args.mean.split(",")
    mean = []
    for mean_v in mean_str:
        mean.append((float)(mean_v))

    target_module_list = []
    hooks_str = args.hooks.split(",")
    for hook in hooks_str:
        target_module_list.append(modules_map[hook])


    bfp_quant(model_name = args.model_name, dataset_dir = args.dataset_dir, num_classes = args.num_classes, gpus = gpus, 
        mantisa_bit = args.mantisa_bit, exp_bit = args.exp_bit, num_bins = args.num_bins, eps = args.eps,
        batch_size = args.batch_size, num_workers = args.num_workers, num_examples = args.num_examples, std=std, mean=mean,
        resize=args.resize, crop=args.crop, exp_act=args.exp_act, bfp_act_chnl=args.bfp_act_chnl, 
        bfp_weight_chnl=args.bfp_weight_chnl, bfp_quant=args.bfp_quant, target_module_list=target_module_list,
        act_bins_factor=args.act_bins_factor, fc_bins_factor=args.fc_bins_factor, is_online=args.is_online)




