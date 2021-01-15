# Internal
from models import model_factory_3d
from lib import Stat_Collector
from lib import Utils
from models import inceptionv4
from lib.dataset import VideoDataset
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
from torch.utils.data import DataLoader
import random

random.seed(666)
torch.manual_seed(666)
np.random.seed(666)
writer = SummaryWriter("./tensorboard/statistics")

modules_map = {  "Linear" : nn.Linear,
                 "BatchNorm3d" : nn.BatchNorm3d,
                 "Conv3d" :  nn.Conv3d
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

    
    train_dataloader = DataLoader(VideoDataset(dataset='ucf101', split='train',clip_len=16, model_name=model_name), batch_size=batch_size, shuffle=True, 
                            num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset='ucf101', split='val',  clip_len=16, model_name=model_name), batch_size=num_examples, num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(dataset='ucf101', split='test', clip_len=16, model_name=model_name), batch_size=batch_size, num_workers=4)

    # # for collect intermediate data use
    # collect_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(resize),
    #         transforms.CenterCrop(crop),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=num_examples, shuffle=False,
    #     num_workers=num_workers, pin_memory=True)
    # # for validate the bfp model use
    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(resize),
    #         transforms.CenterCrop(crop),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=batch_size, shuffle=False,
    #     num_workers=num_workers, pin_memory=True)

    exp_act_list = None 
    lq_model, weight_exp_list = model_factory_3d.get_network(model_name, pretrained=True, bfp=(bfp_quant==1), group=bfp_weight_chnl, mantisa_bit=mantisa_bit, 
                exp_bit=exp_bit, opt_exp_act_list=exp_act_list, is_online=is_online, exp_act=exp_act)

    lq_model.eval()
    lq_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    lq_model.fuse_model()
    lq_model_prepared = torch.quantization.prepare_qat(lq_model)
    #torch.cuda.empty_cache() 

    inpt_fp32 = torch.randn(1, 3, 16, 112, 112)
    lq_model_prepared(inpt_fp32)
    lq_model_8bit = torch.quantization.convert(lq_model_prepared)

    logging.info("Evaluating linear-quant model....")
    correct = 0
    total = 0

    lq_model_8bit.cuda()
    lq_model_8bit.eval()
    with torch.no_grad():
        for i_batch, (images, lables) in enumerate(test_dataloader):
            images = images.cuda()
            outputs = lq_model_8bit(images)
            #outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(probs, 1)
            predicted = predicted.cpu()
            total += lables.size(0)
            correct += (predicted == lables).sum().item()
            logging.info("Current images: %d" % (total))
            #if (total > 2000):
            #    break
    logging.info("Total: %d, Accuracy: %f " % (total, float(correct / total)))
    logging.info("Floating conv weight and fc(act and weight), act bins_factor is %d,fc bins_factor is %d, exp_opt for act is %s, act group is %d"%(act_bins_factor, fc_bins_factor, exp_act, bfp_act_chnl))    
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




