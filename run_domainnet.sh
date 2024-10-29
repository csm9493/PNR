
########### Step 1 : Continual Self-Supervised Learning ###########
# info for Wandb
PROJECT=domainnet-pretrain ## Please change this to YOUR_PROJECT_NAME
ENTITY=sungmin-cha ## Please change this to YOUR_ENTITY

# Other info
GPU_NUM=0,1
DATA_DIR=/home/compu/sungmin/dataset/domainnet ## Please change this to YOUR_DOMAINNET_PATH
SEED=5

# MocoV2Plus + PNR
NAME=mocov2plus-pnr-domainnet-6T-domain PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/domainnet/mocov2plus_pnr.sh
# BYOL + PNR
NAME=byol-pnr-domainnet-6T-domain PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/domainnet/byol_pnr.sh
# VICReg + PNR
NAME=vicreg-pnr-domainnet-6T-domain PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/domainnet/vicreg_pnr.sh
# SimCLR + PNR
NAME=simclr-pnr-domainnet-6T-domain PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/domainnet/simclr_pnr.sh
# Barlow + PNR
NAME=barlow-pnr-domainnet-6T-domain PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/domainnet/barlow_pnr.sh

################################################################################################################################################

# MocoV2Plus + PNR (Load a task 0 model and then train the model starting from task 1)
PRETRAINED_PATH=PRETRAINED_TASK0_MODEL_PATH ## Please change this to YOUR_PRETRAINED_TASK0_MODEL_PATH
NAME=mocov2plus-pnr-domainnet-6T-domain-load PRETRAINED_PATH=$PRETRAINED_PATH PROJECT=$PROJECT ENTITY=$ENTITY SEED=$SEED DATA_DIR=$DATA_DIR CUDA_VISIBLE_DEVICES=$GPU_NUM python job_launcher.py --script bash_files/continual/domainnet/mocov2plus_pnr_load_task0.sh 

################################################################################################################################################

########## Step 2 : Linear Evaluation ###########
# info for Wandb
PROJECT=domainnet-linear ## Please change this to YOUR_PROJECT_NAME
ENTITY=sungmin-cha ## Please change this to YOUR_ENTITY

# Other info
GPU_NUM=0
DATA_DIR=/home/compu/sungmin/dataset/domainnet ## Please change this to YOUR_DOMAINNET_PATH
SEED=5

# MocoV2Plus + PNR
PRETRAINED_PATH=PRETRAINED_MODEL_PATH ## Please change this to YOUR_PPRETRAINED_MODEL_PATH
NAME=mocov2plus-pnr-domainnet-6T-domain PROJECT=$PROJECT ENTITY=$ENTITY GPU_NUM=$GPU_NUM SEED=$SEED PRETRAINED_PATH=$PRETRAINED_PATH DATA_DIR=$DATA_DIR bash bash_files/linear/domainnet/domain/mocov2plus_linear.sh 

# Byol + PNR
PRETRAINED_PATH=PRETRAINED_MODEL_PATH ## Please change this to YOUR_PPRETRAINED_MODEL_PATH
NAME=byol-pnr-domainnet-6T-domain PROJECT=$PROJECT ENTITY=$ENTITY GPU_NUM=$GPU_NUM SEED=$SEED PRETRAINED_PATH=$PRETRAINED_PATH DATA_DIR=$DATA_DIR bash bash_files/linear/domainnet/domain/byol_linear.sh 

# VICReg + PNR
PRETRAINED_PATH=PRETRAINED_MODEL_PATH ## Please change this to YOUR_PPRETRAINED_MODEL_PATH
NAME=vicreg-pnr-domainnet-6T-domain PROJECT=$PROJECT ENTITY=$ENTITY GPU_NUM=$GPU_NUM SEED=$SEED PRETRAINED_PATH=$PRETRAINED_PATH DATA_DIR=$DATA_DIR bash bash_files/linear/domainnet/domain/vicreg_linear.sh 

# SimCLR + PNR
PRETRAINED_PATH=PRETRAINED_MODEL_PATH ## Please change this to YOUR_PPRETRAINED_MODEL_PATH
NAME=simclr-pnr-domainnet-6T-domain PROJECT=$PROJECT ENTITY=$ENTITY GPU_NUM=$GPU_NUM SEED=$SEED PRETRAINED_PATH=$PRETRAINED_PATH DATA_DIR=$DATA_DIR bash bash_files/linear/domainnet/domain/simclr_linear.sh 

# Barlow + PNR
PRETRAINED_PATH=PRETRAINED_MODEL_PATH ## Please change this to YOUR_PPRETRAINED_MODEL_PATH
NAME=barlow-pnr-domainnet-6T-domain PROJECT=$PROJECT ENTITY=$ENTITY GPU_NUM=$GPU_NUM SEED=$SEED PRETRAINED_PATH=$PRETRAINED_PATH DATA_DIR=$DATA_DIR bash bash_files/linear/domainnet/domain/barlow_linear.sh 
