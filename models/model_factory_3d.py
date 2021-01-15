from models import c3d
from models import r3d
from models import res3dnet
from lib.BFPConvertor import BFPConvertor_3D
from benchmark import c3d as br_c3d
from benchmark import r3d as br_r3d

models_map = {  "c3d" : c3d.c3d,
                "lq_c3d" : c3d.c3d_lq,
                "c3d_bfp" : c3d.c3d_bfp,
                "br_c3d_bfp" : br_c3d.c3d_bfp,
                "r3d_18" : r3d.r3d_18,
                "r3d_18_bfp" : r3d.r3d_18_bfp,
                "br_r3d_18_bfp" : br_r3d.r3d_18_bfp,
                "r3d_34" : r3d.r3d_34,
                "r3d_34_bfp" : r3d.r3d_34_bfp,
                "br_r3d_34_bfp" : br_r3d.r3d_34_bfp
}

def get_network(model_name, pretrained=True, bfp=False, group=1, mantisa_bit=8, exp_bit=8, opt_exp_act_list=None, is_online=False, exp_act='kl'):
    if (bfp):
        bfp_model_name = model_name + "_bfp"
        if is_online:
            bfp_model_name = "br_" + bfp_model_name
        assert opt_exp_act_list != None, "When construct the bfp model, the opt_exp_act should be Non-empty"
        golden_model = models_map[model_name](101, pretrained=True)
        c3d_converter = BFPConvertor_3D(mantisa_bit, exp_bit)
        bfp_model = models_map[bfp_model_name](num_classes=101, pretrained=True, exp_bit=exp_bit, mantisa_bit=mantisa_bit, opt_exp_act_list=opt_exp_act_list)
        conv_isbias = True if ("c3d" in model_name) else False
        bfp_model, weight_exp_list = c3d_converter(golden_model, bfp_model, group, conv_isbias=conv_isbias, is_kl=True, exp_act=exp_act)
        return bfp_model, weight_exp_list
    else:
        return models_map[model_name](num_classes=101, pretrained=True), None
