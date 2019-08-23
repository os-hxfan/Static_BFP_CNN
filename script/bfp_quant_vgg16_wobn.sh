python -m tools.bfp_quant --model_name vgg16 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 4 \
--batch_size 32 \
--num_workers 8 \
--num_classes 1000 \
--gpus 3 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act kl \
--bfp_act_chnl 256 \
--bfp_weight_chnl 128 \
--bfp_quant 1 \
--num_examples 40 \
--hooks Conv2d,Linear \
--act_bins_factor 6

