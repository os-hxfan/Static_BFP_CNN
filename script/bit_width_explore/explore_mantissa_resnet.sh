python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 8 \
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
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m8_e8.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 9 \
--exp_bit 8 \
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
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m9_e8.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 10 \
--exp_bit 8 \
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
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m10_e8.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 11 \
--exp_bit 8 \
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
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m11_e8.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 12 \
--exp_bit 8 \
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
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m12_e8.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 7 \
--exp_bit 8 \
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
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m7_e8.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 6 \
--exp_bit 8 \
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
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m6_e8.txt

python -m tools.bfp_quant --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 5 \
--exp_bit 8 \
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
--fc_bins_factor 5 2>&1 | tee log/bit_width_explore/resnet50/resnet50_m5_e8.txt