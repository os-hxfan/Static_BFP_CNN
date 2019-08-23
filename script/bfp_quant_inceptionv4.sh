python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 4 \
--batch_size 480 \
--num_workers 8 \
--num_classes 1000 \
--gpus 1,2,3,4 \
--std 0.5,0.5,0.5 \
--mean 0.5,0.5,0.5 \
--resize 299 \
--crop 299 \
--exp_act kl \
--bfp_act_chnl -1 \
--bfp_weight_chnl 64 \
--bfp_quant 1 \
--act_bins_factor 5 \
--hooks BatchNorm2d,Linear,Mixed_4a,Mixed_5a,Inception_A,\
Reduction_A,Inception_B,Reduction_B,Inception_C 

