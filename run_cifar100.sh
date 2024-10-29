
# Info for Wandb
PROJECT=cifar100 ## Please change this to YOUR_PROJECT_NAME
ENTITY=sungmin-cha ## Please change this to YOUR_ENTITY

# Other info
GPU_NUM=0
DATA_DIR=/home/compu/sungmin/dataset ## Please change this to YOUR_DATASET_PATH
SEED=5
NUM_TASKS=5

# MocoV2Plus + PNR
NAME=mocov2plus-pnr-cifar100-5T PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/cifar/mocov2plus_pnr.sh 
# BYOL + PNR
NAME=byol-pnr-cifar100-5T PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/cifar/byol_pnr.sh 
# VICReg + PNR
NAME=vicreg-pnr-cifar100-5T PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/cifar/vicreg_pnr.sh 
# SimCLR + PNR
NAME=simclr-pnr-cifar100-5T PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/cifar/simclr_pnr.sh 
# Barlow + PNR
NAME=barlow-pnr-cifar100-5T PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/cifar/barlow_pnr.sh 

# MocoV2Plus + PNR (Load a task 0 model and then train the model starting from task 1)
PRETRAINED_PATH=PRETRAINED_TASK0_MODEL_PATH
NAME=mocov2plus-pnr-cifar100-5T-load PRETRAINED_PATH=$PRETRAINED_PATH PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED NUM_TASKS=$NUM_TASKS DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/cifar/mocov2plus_pnr_load_task0.sh 