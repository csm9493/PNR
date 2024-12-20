python3 main_continual.py \
    --dataset cifar100 \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --seed $SEED \
    --split_strategy class \
    --task_idx 0 \
    --num_tasks $NUM_TASKS \
    --max_epochs 500 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --num_workers 3 \
    --min_scale 0.2 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --solarization_prob 0.1 \
    --gaussian_prob 0.0 0.0 \
    --name $NAME \
    --project $PROJECT \
    --entity $ENTITY \
    --wandb \
    --save_checkpoint \
    --method vicreg \
    --proj_hidden_dim 2048 \
    --output_dim 2048 \
    --sim_loss_weight 25.0 \
    --var_loss_weight 25.0 \
    --cov_loss_weight 1.0 \
    --negative_lamb 23.0 \
    --distiller vicreg_pnr 
