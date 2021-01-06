python -m tools.bfp_quant_3d --model_name c3d \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 4 \
--batch_size 8 \
--num_workers 8 \
--num_classes 101 \
--gpus 2 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act kl \
--bfp_act_chnl 1 \
--bfp_weight_chnl 1 \
--bfp_quant 1 \
--num_examples 10 \
--hooks BatchNorm2d,Linear \
--act_bins_factor 6

