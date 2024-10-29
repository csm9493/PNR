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
    --min_scale 0.2 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --solarization_prob 0.1 \
    --dali \
    --check_val_every_n_epoch 9999 \
    --name $NAME \
    --save_checkpoint \
    --wandb \
    --project $PROEJCT \
    --entity $ENTITY \
    --method vicreg \
    --proj_hidden_dim 2048 \
    --output_dim 2048 \
    --sim_loss_weight 25.0 \
    --var_loss_weight 25.0 \
    --cov_loss_weight 1.0 \
    --negative_lamb 5.0 \
    --distiller vicreg_pnr 
