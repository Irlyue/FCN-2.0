function train_n_epochs
{
    python train.py \
        --lr=$LR \
        --lrp=$LRP \
        --batch_size=$TRAIN_BATCH \
        --n_epochs=$1 \
        --backbone=$BACKBONE \
        --ckpt_for_backbone=/tmp/$RESNET_CKPT \
        --image_size=$IMAGE_SIZE $IMAGE_SIZE \
        --data=voc2012-train-Segmentation \
        --reg=$REG \
        --solver=$SOLVER \
        --model_dir=$MODEL_DIR \
        --gpu_id=$TRAIN_GPU_ID

}

function pre_run_op
{
    if [ -e /tmp/$RESNET_CKPT ]
    then
        echo "File $RESNET_CKPT exists"
    else
        echo "Extracting $RESNET_CKPT"
        tar -zxf ~/models/weights/resnet_v1_101_2016_08_28.tar.gz -C /tmp/
    fi

    if [ -e $MODEL_DIR ]
    then
        echo "Removing existing model directory $MODEL_DIR"
        rm -rf $MODEL_DIR
    fi
}

function eval_on_gpu
{
    python pys/eval_every_new_ckpt.py \
        --batch_size=$EVAL_BATCH \
        --image_size=$IMAGE_SIZE $IMAGE_SIZE \
        --reg=$REG \
        --data=$1 \
        --model_dir=$MODEL_DIR \
        --gpu_id=$2
}

############################################

###############
#  Setting    #
###############
BACKBONE="resnet_v1_101"
RESNET_CKPT="resnet_v1_101.ckpt"
IMAGE_SIZE=448
REG=1e-4
MODEL_DIR="/tmp/fcn"
TRAIN_GPU_ID=2
EVAL_GPU_ID=0

TRAIN_BATCH=16
EVAL_BATCH=16


##################
# Running        #
##################
cd ..
trap "exit" INT


if [ "$1" == "train" ]; then
    echo "In training"
    pre_run_op
    LR=1e-3
    LRP=0.005

    train_n_epochs 10

elif [ "$1" == "eval" ]; then
    echo "In evaluation"
    #CUDA_VISIBLE_DEVICES=1 eval_on_gpu voc2012-val-SegMain $EVAL_GPU_ID
    eval_on_gpu voc2012-val-Segmentation $EVAL_GPU_ID
elif [ "$1" == "eval_once" ]; then
    eval_once voc2012-val-Segmentation $EVAL_GPU_ID
else
    echo "Parameter '$1' not allowed"
fi