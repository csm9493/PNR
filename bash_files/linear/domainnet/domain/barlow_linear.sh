# all datasets
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --seed $SEED \
    --split_strategy domain \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name ${NAME}_domainnet_all-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --wandb \
    --project $PROJECT \
    --entity $ENTITY \
    --save_checkpoint

# quickdraw
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --seed $SEED \
    --split_strategy domain \
    --domain quickdraw \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name ${NAME}_domain_quickdraw-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --wandb \
    --project $PROJECT \
    --entity $ENTITY \
    --save_checkpoint

# clipart
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --seed $SEED \
    --split_strategy domain \
    --domain clipart \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name ${NAME}_domain_clipart-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --wandb \
    --project $PROJECT \
    --entity $ENTITY \
    --save_checkpoint

# infograph
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --seed $SEED \
    --split_strategy domain \
    --domain infograph \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name ${NAME}_domain_infograph-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --wandb \
    --project $PROJECT \
    --entity $ENTITY \
    --save_checkpoint

# painting
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --seed $SEED \
    --split_strategy domain \
    --domain painting \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name ${NAME}_domain_painting-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --wandb \
    --project $PROJECT \
    --entity $ENTITY \
    --save_checkpoint

# real
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --seed $SEED \
    --split_strategy domain \
    --domain real \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name ${NAME}_domain_real-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --wandb \
    --project $PROJECT \
    --entity $ENTITY \
    --save_checkpoint

# sketch
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_linear.py \
    --dataset domainnet \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --seed $SEED \
    --split_strategy domain \
    --domain sketch \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name ${NAME}_domain_sketch-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --wandb \
    --project $PROJECT \
    --entity $ENTITY \
    --save_checkpoint

