CUDA_VISIBLE_DEVICES=$GPU_NUM python3 main_linear.py \
    --dataset imagenet100 \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --seed $SEED \
    --train_dir $TRAIN_DIR \
    --val_dir $VAL_DIR \
    --split_strategy class \
    --num_tasks 5 \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 3.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 8 \
    --dali \
    --name $NAME \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project $PROEJCT \
    --entity $ENTITY \
    --wandb \
    --save_checkpoint