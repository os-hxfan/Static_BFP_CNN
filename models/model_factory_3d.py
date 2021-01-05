from models import c3d

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
        golden_model = c3d.c3d(101, pretrained=True)
        c3d_converter = BFPConvertor_3D(mantisa_bit, exp_bit)
        bfp_model = c3d.c3d_bfp(num_classes=101, pretrained=True, exp_bit=exp_bit, mantisa_bit=mantisa_bit, opt_exp_act_list=opt_exp_act_list)
        bfp_model, weight_exp_list = c3d_converter(golden_model, bfp_model, group, is_kl=True)
        return bfp_model
    else:
        return c3d.c3d(num_classes=101, pretrained=True)
