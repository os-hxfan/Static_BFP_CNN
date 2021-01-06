import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
import numpy as np

# Internal
from lib.BFPConvertor import BFPConvertor_3D
from lib import BFPActivation
from lib import BFPFullyConnet
# PyTorch
import torch
import torch.nn as nn
import torchvision

class c3d(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        super(c3d, self).__init__()
        print ("Construct original C3D model")
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4a = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4b = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5a = nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5b = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))
        x = self.pool4(x)

        x = self.relu(self.bn5a(self.conv5a(x)))
        x = self.relu(self.bn5b(self.conv5b(x)))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        #p_dict = torch.load("/mnt/ccnas2/bdp/hf17/TCAD_3DCNNs/c3d-pretrained.pth")
        p_dict = torch.load("/mnt/ccnas2/bdp/hf17/TCAD_3DCNNs/C3D-ucf101_epoch-99.pth.tar")
        print ("Loading from pretrained models")
        self.load_state_dict(p_dict['state_dict'])
        s_dict = self.state_dict()
        '''
        for name in p_dict:
            if name in s_dict:
                self.state_dict()[name] = p_dict
                print ("Load from layer:", name)
            else:
                print ("not found layer:", name)
        '''
        #self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


class c3d_bfp(nn.Module):
    """
    The C3D network with BFP quantization.
    """

    def __init__(self, num_classes, pretrained=False, exp_bit=8, mantisa_bit=8, opt_exp_act_list=None):
        super(c3d_bfp, self).__init__()
        print ("Construct BFP C3D model")
        self.exp_bit = exp_bit
        self.mantisa_bit = mantisa_bit
        self.opt_exp_act_list = opt_exp_act_list

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3a = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3b = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4a = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4b = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5a = nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn5b = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()


    def forward(self, x):

        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[0], is_3d=True)
        
        x = self.bn1(self.conv1(x))

        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[1], is_3d=True)

        x = self.relu(x)
        x = self.pool1(x)

        x = self.bn2(self.conv2(x))

        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[2], is_3d=True)

        x = self.relu(x)
        x = self.pool2(x)

        x = self.bn3a(self.conv3a(x))

        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[3], is_3d=True)

        x = self.relu(x)
        x = self.bn3b(self.conv3b(x))

        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[4], is_3d=True)

        x = self.relu(x)
        x = self.pool3(x)

        x = self.bn4a(self.conv4a(x))

        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[5], is_3d=True)

        x = self.relu(x)
        x = self.bn4b(self.conv4b(x))

        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[6], is_3d=True)

        x = self.relu(x)
        x = self.pool4(x)

        x = self.bn5a(self.conv5a(x))

        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[7], is_3d=True)

        x = self.relu(x)
        x = self.bn5b(self.conv5b(x))

        x = BFPActivation.transform_activation_offline(x, self.exp_bit, self.mantisa_bit,
                                                         self.opt_exp_act_list[8], is_3d=True)

        x = self.relu(x)
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.fc6(x)
        x = BFPFullyConnet.transform_fc_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[9])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = BFPFullyConnet.transform_fc_offline(x, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[10])

        x = self.relu(x)
        x = self.dropout(x)

        logits = self.fc8(x)
        #print (logits.shape)
        x = BFPFullyConnet.transform_fc_offline(logits, self.exp_bit, self.mantisa_bit, self.opt_exp_act_list[11])

        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load("/mnt/ccnas2/bdp/hf17/TCAD_3DCNNs/c3d-pretrained.pth")
        p_dict = p_dict[state_dict]
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = c3d(num_classes=101, pretrained=True)

    outputs = net.forward(inputs)
    print(outputs.size())
