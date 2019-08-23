python -m tools.eval --model_name resnet50 \
--dataset_dir /mnt/ccnas2/bdp/hf17/Datasets/Imagenet12/ \
--batch_size 512 \
--num_workers 8 \
--num_classes 1000 \
--gpus 1,2,3
