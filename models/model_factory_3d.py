from models import c3d
from lib.BFPConvertor import BFPConvertor_3D

def get_network(model_name, pretrained=True, bfp=False, group=1, mantisa_bit=8, exp_bit=8, opt_exp_act_list=None):
    if (bfp):
        assert opt_exp_act_list != None, "When construct the bfp model, the opt_exp_act should be Non-empty"
        golden_model = c3d.c3d(101, pretrained=True)
        c3d_converter = BFPConvertor_3D(mantisa_bit, exp_bit)
        bfp_model = c3d.c3d_bfp(num_classes=101, pretrained=True, exp_bit=exp_bit, mantisa_bit=mantisa_bit, opt_exp_act_list=opt_exp_act_list)
        bfp_model, weight_exp_list = c3d_converter(golden_model, bfp_model, group, conv_isbias=True, is_kl=True)
        return bfp_model, weight_exp_list
    else:
        return c3d.c3d(num_classes=101, pretrained=True), None
