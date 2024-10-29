
########### Step 1 : Continual Self-Supervised Learning ###########
# info for Wandb
PROJECT=imagenet-100-pretrain ## Please change this to YOUR_PROJECT_NAME
ENTITY=sungmin-cha ## Please change this to YOUR_ENTITY

# Other info
GPU_NUM=0,1
DATA_DIR=/home/compu/sungmin/dataset ## Please change this to YOUR_DATASET_PATH
TRAIN_DIR=/home/compu/sungmin/dataset/Imagenet100/train ## Please change this to YOUR_IMAGENET100_TRAIN_PATH
VAL_DIR=/home/compu/sungmin/dataset/Imagenet100/val ## Please change this to YOUR_IMAGENET100_VAL_PATH
SEED=5
NUM_TASKS=5

# MocoV2Plus + PNR
NAME=mocov2plus-pnr-imagenet100-5T-data PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR TRAIN_DIR=$TRAIN_DIR VAL_DIR=$VAL_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/imagenet-100/data/mocov2plus_pnr.sh
# BYOL + PNR
NAME=byol-pnr-imagenet100-5T-data PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR TRAIN_DIR=$TRAIN_DIR VAL_DIR=$VAL_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/imagenet-100/data/byol_pnr.sh
# VICReg + PNR
NAME=vicreg-pnr-imagenet100-5T-data PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR TRAIN_DIR=$TRAIN_DIR VAL_DIR=$VAL_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/imagenet-100/data/vicreg_pnr.sh
# SimCLR + PNR
NAME=simclr-pnr-imagenet100-5T-data PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR TRAIN_DIR=$TRAIN_DIR VAL_DIR=$VAL_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/imagenet-100/data/simclr_pnr.sh
# Barlow + PNR
NAME=barlow-pnr-imagenet100-5T-data PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR TRAIN_DIR=$TRAIN_DIR VAL_DIR=$VAL_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/imagenet-100/data/barlow_pnr.sh

################################################################################################################################################

# MocoV2Plus + PNR (Load a task 0 model and then train the model starting from task 1)
PRETRAINED_PATH=PRETRAINED_TASK0_MODEL_PATH ## Please change this to YOUR_PRETRAINED_TASK0_MODEL_PATH
NAME=mocov2plus-pnr-imagenet100-5T-data-load PRETRAINED_PATH=$PRETRAINED_PATH PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/imagenet-100/data/mocov2plus_pnr_load_task0.sh 

################################################################################################################################################

########## Step 2 : Linear Evaluation ###########
# info for Wandb
PROJECT=imagenet-100-linear ## Please change this to YOUR_PROJECT_NAME
ENTITY=sungmin-cha ## Please change this to YOUR_ENTITY

# Other info
GPU_NUM=0
DATA_DIR=/home/compu/sungmin/dataset ## Please change this to YOUR_DATASET_PATH
TRAIN_DIR=/home/compu/sungmin/dataset/Imagenet100/train ## Please change this to YOUR_IMAGENET100_TRAIN_PATH
VAL_DIR=/home/compu/sungmin/dataset/Imagenet100/val ## Please change this to YOUR_IMAGENET100_VAL_PATH
SEED=5

# MocoV2Plus + PNR
PRETRAINED_PATH=PRETRAINED_MODEL_PATH
NAME=mocov2plus-pnr-imagenet100-5T-data-linear-eval PROJECT=$PROJECT ENTITY=$ENTITY GPU_NUM=$GPU_NUM SEED=$SEED PRETRAINED_PATH=$PRETRAINED_PATH DATA_DIR=$DATA_DIR TRAIN_DIR=$TRAIN_DIR VAL_DIR=$VAL_DIR bash bash_files/linear/imagenet-100/data/mocov2plus_linear.sh 

# Byol + PNR
PRETRAINED_PATH=PRETRAINED_MODEL_PATH
NAME=byol-pnr-imagenet100-5T-data-linear-eval PROJECT=$PROJECT ENTITY=$ENTITY GPU_NUM=$GPU_NUM SEED=$SEED PRETRAINED_PATH=$PRETRAINED_PATH DATA_DIR=$DATA_DIR TRAIN_DIR=$TRAIN_DIR VAL_DIR=$VAL_DIR bash bash_files/linear/imagenet-100/data/byol_linear.sh 

# VICReg + PNR
PRETRAINED_PATH=PRETRAINED_MODEL_PATH
NAME=vicreg-pnr-imagenet100-5T-data-linear-eval PROJECT=$PROJECT ENTITY=$ENTITY GPU_NUM=$GPU_NUM SEED=$SEED PRETRAINED_PATH=$PRETRAINED_PATH DATA_DIR=$DATA_DIR TRAIN_DIR=$TRAIN_DIR VAL_DIR=$VAL_DIR bash bash_files/linear/imagenet-100/data/vicreg_linear.sh 

# SimCLR + PNR
PRETRAINED_PATH=PRETRAINED_MODEL_PATH
NAME=simclr-pnr-imagenet100-5T-data-linear-eval PROJECT=$PROJECT ENTITY=$ENTITY GPU_NUM=$GPU_NUM SEED=$SEED PRETRAINED_PATH=$PRETRAINED_PATH DATA_DIR=$DATA_DIR TRAIN_DIR=$TRAIN_DIR VAL_DIR=$VAL_DIR bash bash_files/linear/imagenet-100/data/simclr_linear.sh 

# Barlow + PNR
PRETRAINED_PATH=PRETRAINED_MODEL_PATH
NAME=barlow-pnr-imagenet100-5T-data-linear-eval PROJECT=$PROJECT ENTITY=$ENTITY GPU_NUM=$GPU_NUM SEED=$SEED PRETRAINED_PATH=$PRETRAINED_PATH DATA_DIR=$DATA_DIR TRAIN_DIR=$TRAIN_DIR VAL_DIR=$VAL_DIR bash bash_files/linear/imagenet-100/data/barlow_linear.sh 
