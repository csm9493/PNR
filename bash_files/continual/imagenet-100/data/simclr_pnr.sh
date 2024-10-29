python3 main_continual.py \
    --dataset imagenet100 \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --seed $SEED \
    --train_dir $TRAIN_DIR \
    --val_dir $VAL_DIR \
    --split_strategy data \
    --max_epochs 400 \
    --num_tasks $NUM_TASKS \
    --task_idx 0 \
    --gpus 0,1 \
    --accelerator ddp \
    --sync_batchnorm \
    --num_workers 5 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --dali \
    --check_val_every_n_epoch 9999 \
    --name $NAME \
    --save_checkpoint \
    --wandb \
    --project $PROEJCT \
    --entity $ENTITY \
    --method simclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --distiller simclr_pnr 
