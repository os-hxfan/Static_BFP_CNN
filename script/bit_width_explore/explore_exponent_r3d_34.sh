python -m tools.bfp_quant_3d --model_name r3d_34 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 9 \
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
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/r3d_34/r3d_34_m8_e9.txt


python -m tools.bfp_quant_3d --model_name r3d_34 \
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
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/r3d_34/r3d_34_m8_e8.txt

python -m tools.bfp_quant_3d --model_name r3d_34 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 7 \
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
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/r3d_34/r3d_34_m8_e7.txt

python -m tools.bfp_quant_3d --model_name r3d_34 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 6 \
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
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/r3d_34/r3d_34_m8_e6.txt

python -m tools.bfp_quant_3d --model_name r3d_34 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 5 \
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
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/r3d_34/r3d_34_m8_e5.txt

python -m tools.bfp_quant_3d --model_name r3d_34 \
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
--exp_act max \
--bfp_act_chnl -1 \
--bfp_weight_chnl -1 \
--bfp_quant 1 \
--num_examples 10 \
--hooks BatchNorm3d,Linear \
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/r3d_34/r3d_34_m8_e4.txt

python -m tools.bfp_quant_3d --model_name r3d_34 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 3 \
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
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/r3d_34/r3d_34_m8_e3.txt

python -m tools.bfp_quant_3d --model_name r3d_34 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--mantisa_bit 8 \
--exp_bit 2 \
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
--act_bins_factor 6  2>&1 | tee log/bit_width_explore/r3d_34/r3d_34_m8_e2.txt