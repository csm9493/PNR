python3 main_continual.py \
    --dataset imagenet100 \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --seed $SEED \
    --train_dir $TRAIN_DIR \
    --val_dir $VAL_DIR \
    --split_strategy class \
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
    --scheduler warmup_cosine \
    --lr 0.6 \
    --weight_decay 1e-4 \
    --classifier_lr 0.3 \
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
    --wandb \
    --project $PROJECT \
    --entity $ENTITY \
    --save_checkpoint \
    --method mocov2plus \
    --proj_hidden_dim 2048 \
    --queue_size 65536 \
    --temperature 0.2 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --momentum_classifier \
    --distiller mocov2plus_pnr \
    --distill_temperature 0.2 \
    --pretrained_model $PRETRAINED_PATH


