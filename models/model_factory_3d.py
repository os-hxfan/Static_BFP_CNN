from models import c3d
from models import r3d
from models import res3dnet
from lib.BFPConvertor import BFPConvertor_3D

models_map = {  "c3d" : c3d.c3d,
                "c3d_bfp" : c3d.c3d_bfp,
                "r3d" : r3d.r3d,
                "r3d_bfp" : r3d.r3d_bfp
}

def get_network(model_name, pretrained=True, bfp=False, group=1, mantisa_bit=8, exp_bit=8, opt_exp_act_list=None):
    if (bfp):
        bfp_model_name = model_name + "_bfp"
        assert opt_exp_act_list != None, "When construct the bfp model, the opt_exp_act should be Non-empty"
        golden_model = models_map[model_name](101, pretrained=True)
        c3d_converter = BFPConvertor_3D(mantisa_bit, exp_bit)
        bfp_model = models_map[bfp_model_name](num_classes=101, pretrained=True, exp_bit=exp_bit, mantisa_bit=mantisa_bit, opt_exp_act_list=opt_exp_act_list)
        conv_isbias = True if model_name=="c3d" else False
        bfp_model, weight_exp_list = c3d_converter(golden_model, bfp_model, group, conv_isbias=conv_isbias, is_kl=True)
        return bfp_model, weight_exp_list
    else:
        return models_map[model_name](num_classes=101, pretrained=True), None
