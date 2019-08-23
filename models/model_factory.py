from models import resnet
from models import inceptionv4
from models import mobilenetv2
from models import vgg
import benchmark.resnet as br_resnet
import benchmark.mobilenetv2 as br_mobilenetv2
import benchmark.inceptionv4 as br_inceptionv4
import benchmark.vgg as br_vgg

models_map = {  "resnet34" : resnet.resnet34,
                "resnet50" : resnet.resnet50,
                "resnet100" :  resnet.resnet50,
                "inceptionv4" : inceptionv4.inceptionv4,
                "mobilenetv2" : mobilenetv2.block_mobilenet,
                "vgg16" : vgg.block_vgg16,
                "br_resnet50" : br_resnet.resnet50,
                "br_mobilenetv2" : br_mobilenetv2.block_mobilenet,
                "br_vgg16" : br_vgg.block_vgg16,
                "br_inceptionv4" : br_inceptionv4.inceptionv4
}

def get_network(model_name, pretrained=True, bfp=False, group=1, mantisa_bit=8, exp_bit=8, opt_exp_act_list=None):
    if (bfp):
        assert opt_exp_act_list != None, "When construct the bfp model, the opt_exp_act should be Non-empty"
    return models_map[model_name](pretrained=pretrained, bfp=bfp, group=group, mantisa_bit=mantisa_bit,
                 exp_bit=exp_bit, opt_exp_act_list=opt_exp_act_list)
