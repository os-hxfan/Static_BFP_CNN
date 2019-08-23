python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 12 \
--exp_bit 8 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m12_e8.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 11 \
--exp_bit 8 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m11_e8.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 10 \
--exp_bit 8 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m10_e8.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 9 \
--exp_bit 8 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m9_e8.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 8 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m8_e8.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 7 \
--exp_bit 8 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m7_e8.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 6 \
--exp_bit 8 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m6_e8.txt

python -m tools.bfp_quant --model_name inceptionv4 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 5 \
--exp_bit 8 \
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
Reduction_A,Inception_B,Reduction_B,Inception_C 2>&1 | tee log/bit_width_explore/inceptionv4/inceptionv4_m5_e8.txt