# Internal
from models import model_factory

# The Basic Library
import argparse 
import os
import logging
import numpy as np
import sys
import copy
import math

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Other Required Library
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def eval(model_name, dataset_dir, num_classes, gpus, batch_size = 1, num_workers = 2) :
    # Setting up gpu environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)

    # Setting up dataload for evaluation
    valdir = os.path.join(dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    # Loading the model
    model = model_factory.get_network(model_name, pretrained=True)
    model.cuda()
    model.eval()

    correct = 0
    total = 0
    # Running the evaluation
    with torch.no_grad():
        for i_batch, (images, lables) in enumerate(val_loader):
            images = images.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
            total += lables.size(0)
            correct += (predicted == lables).sum().item()
            if (total >= 1000):
                break
    print ("Total: %d, Accuracy: %f " % (total, float(correct / total)))


    print ("Finish the Evaluation")

    # Ploting the distribution
    


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="resnet34", type=str, required=True, help="now only support one of {resnet34,resnet50,resnet101}")
    parser.add_argument("--dataset_dir", default="/dataset/", type=str, required=True, help="The absolute path point to the dataset")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size on each gpu")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers in data loader")
    parser.add_argument("--num_classes", default=1000, type=int, required=True, help="The number of classes")
    parser.add_argument("--gpus", default="0, 1", type=str, required=True, help="GPUs id, separated by comma withougt space, for example: 0,1,2")

    args = parser.parse_args()

    # Split the argument
    gpus = args.gpus.split(",")
    gpus_num = len(gpus)
    logging.info("totally {} gpus are using".format(gpus_num))


    eval(model_name = args.model_name, dataset_dir = args.dataset_dir, num_classes = args.num_classes, gpus = gpus, 
        batch_size = args.batch_size, num_workers = args.num_workers)




