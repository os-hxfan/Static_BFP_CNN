python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 9 \
--num_examples 50 \
--batch_size 64 \
--num_workers 8 \
--num_classes 1000 \
--gpus 5 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act kl \
--bfp_act_chnl -1 \
--bfp_weight_chnl 64 \
--bfp_quant 1 \
--act_bins_factor 4 \
--hooks BatchNorm2d,Linear \
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m8_e9.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 7 \
--num_examples 50 \
--batch_size 64 \
--num_workers 8 \
--num_classes 1000 \
--gpus 5 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act kl \
--bfp_act_chnl -1 \
--bfp_weight_chnl 64 \
--bfp_quant 1 \
--act_bins_factor 4 \
--hooks BatchNorm2d,Linear \
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m8_e7.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 6 \
--num_examples 50 \
--batch_size 64 \
--num_workers 8 \
--num_classes 1000 \
--gpus 5 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act kl \
--bfp_act_chnl -1 \
--bfp_weight_chnl 64 \
--bfp_quant 1 \
--act_bins_factor 4 \
--hooks BatchNorm2d,Linear \
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m8_e6.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 5 \
--num_examples 50 \
--batch_size 64 \
--num_workers 8 \
--num_classes 1000 \
--gpus 5 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act kl \
--bfp_act_chnl -1 \
--bfp_weight_chnl 64 \
--bfp_quant 1 \
--act_bins_factor 4 \
--hooks BatchNorm2d,Linear \
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m8_e5.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 4 \
--num_examples 50 \
--batch_size 64 \
--num_workers 8 \
--num_classes 1000 \
--gpus 5 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act kl \
--bfp_act_chnl -1 \
--bfp_weight_chnl 64 \
--bfp_quant 1 \
--act_bins_factor 4 \
--hooks BatchNorm2d,Linear \
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m8_e4.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 3 \
--num_examples 50 \
--batch_size 64 \
--num_workers 8 \
--num_classes 1000 \
--gpus 5 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act kl \
--bfp_act_chnl -1 \
--bfp_weight_chnl 64 \
--bfp_quant 1 \
--act_bins_factor 4 \
--hooks BatchNorm2d,Linear \
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m8_e3.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 2 \
--num_examples 50 \
--batch_size 64 \
--num_workers 8 \
--num_classes 1000 \
--gpus 5 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act kl \
--bfp_act_chnl -1 \
--bfp_weight_chnl 64 \
--bfp_quant 1 \
--act_bins_factor 4 \
--hooks BatchNorm2d,Linear \
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m8_e2.txt