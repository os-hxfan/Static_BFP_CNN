python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 9 \
--batch_size 200 \
--num_workers 8 \
--num_classes 1000 \
--gpus 1,2 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m8_e9.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 7 \
--batch_size 200 \
--num_workers 8 \
--num_classes 1000 \
--gpus 1,2 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m8_e7.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 6 \
--batch_size 200 \
--num_workers 8 \
--num_classes 1000 \
--gpus 1,2 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m8_e6.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 5 \
--batch_size 200 \
--num_workers 8 \
--num_classes 1000 \
--gpus 1,2 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m8_e5.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 4 \
--batch_size 200 \
--num_workers 8 \
--num_classes 1000 \
--gpus 1,2 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m8_e4.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 3 \
--batch_size 200 \
--num_workers 8 \
--num_classes 1000 \
--gpus 1,2 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m8_e3.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 2 \
--batch_size 200 \
--num_workers 8 \
--num_classes 1000 \
--gpus 1,2 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m8_e2.txt