python -m tools.bfp_quant_3d --model_name c3d \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 12 \
--exp_bit 8 \
--batch_size 8 \
--num_workers 8 \
--num_classes 101 \
--gpus 2 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act max \
--bfp_act_chnl -1 \
--bfp_weight_chnl -1 \
--bfp_quant 1 \
--num_examples 10 \
--hooks BatchNorm3d,Linear \
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/c3d/c3d_m12_e8.txt


python -m tools.bfp_quant_3d --model_name c3d \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 11 \
--exp_bit 8 \
--batch_size 8 \
--num_workers 8 \
--num_classes 101 \
--gpus 2 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act max \
--bfp_act_chnl -1 \
--bfp_weight_chnl -1 \
--bfp_quant 1 \
--num_examples 10 \
--hooks BatchNorm3d,Linear \
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/c3d/c3d_m11_e8.txt

python -m tools.bfp_quant_3d --model_name c3d \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 10 \
--exp_bit 8 \
--batch_size 8 \
--num_workers 8 \
--num_classes 101 \
--gpus 2 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act max \
--bfp_act_chnl -1 \
--bfp_weight_chnl -1 \
--bfp_quant 1 \
--num_examples 10 \
--hooks BatchNorm3d,Linear \
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/c3d/c3d_m10_e8.txt

python -m tools.bfp_quant_3d --model_name c3d \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 9 \
--exp_bit 8 \
--batch_size 8 \
--num_workers 8 \
--num_classes 101 \
--gpus 2 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act max \
--bfp_act_chnl -1 \
--bfp_weight_chnl -1 \
--bfp_quant 1 \
--num_examples 10 \
--hooks BatchNorm3d,Linear \
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/c3d/c3d_m9_e8.txt

python -m tools.bfp_quant_3d --model_name c3d \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 8 \
--batch_size 8 \
--num_workers 8 \
--num_classes 101 \
--gpus 2 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act max \
--bfp_act_chnl -1 \
--bfp_weight_chnl -1 \
--bfp_quant 1 \
--num_examples 10 \
--hooks BatchNorm3d,Linear \
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/c3d/c3d_m8_e8.txt


python -m tools.bfp_quant_3d --model_name c3d \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 7 \
--exp_bit 8 \
--batch_size 8 \
--num_workers 8 \
--num_classes 101 \
--gpus 2 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act max \
--bfp_act_chnl -1 \
--bfp_weight_chnl -1 \
--bfp_quant 1 \
--num_examples 10 \
--hooks BatchNorm3d,Linear \
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/c3d/c3d_m7_e8.txt


python -m tools.bfp_quant_3d --model_name c3d \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 6 \
--exp_bit 8 \
--batch_size 8 \
--num_workers 8 \
--num_classes 101 \
--gpus 2 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act max \
--bfp_act_chnl -1 \
--bfp_weight_chnl -1 \
--bfp_quant 1 \
--num_examples 10 \
--hooks BatchNorm3d,Linear \
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/c3d/c3d_m6_e8.txt


python -m tools.bfp_quant_3d --model_name c3d \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 5 \
--exp_bit 8 \
--batch_size 8 \
--num_workers 8 \
--num_classes 101 \
--gpus 2 \
--std 0.229,0.224,0.225 \
--mean 0.485,0.456,0.406 \
--resize 256 \
--crop 224 \
--exp_act max \
--bfp_act_chnl -1 \
--bfp_weight_chnl -1 \
--bfp_quant 1 \
--num_examples 10 \
--hooks BatchNorm3d,Linear \
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/c3d/c3d_m5_e8.txt