function train_n_epochs
{
    python train.py \
        --lr=$LR \
        --lrp=$LRP \
        --batch_size=$TRAIN_BATCH \
        --n_epochs=$1 \
        --backbone=$BACKBONE \
        --ckpt_for_backbone=/tmp/$RESNET_CKPT \
        --image_size $IMAGE_SIZE $IMAGE_SIZE \
        --data=$TRAIN_DATA \
        --reg=$REG \
		--aspp_rates$RATES \
        --solver=$SOLVER \
        --model_dir=$MODEL_DIR \
		--backbone_stride=$BACKBONE_STRIDE \
		--keep_prob=$KEEP_PROB \
		--kernel_size=$KERNEL_SIZE \
        --gpu_id=$TRAIN_GPU_ID

}

function pre_run_op
{
    if [ -e /tmp/$RESNET_CKPT ]
    then
        echo "File $RESNET_CKPT exists"
    else
        echo "Extracting $RESNET_CKPT"
        tar -zxf ~/models/weights/vgg_16_2016_08_28.tar.gz -C /tmp/
    fi

    if [ -e $MODEL_DIR ]
    then
        echo "Removing existing model directory $MODEL_DIR"
        rm -rf $MODEL_DIR
    fi
}

function eval_online
{
    python pys/eval_every_new_ckpt.py \
        --batch_size=$EVAL_BATCH \
        --image_size $IMAGE_SIZE $IMAGE_SIZE \
        --reg=$REG \
        --data=$1 \
		--aspp_rates$RATES \
		--keep_prob=$KEEP_PROB \
		--backbone_stride=$BACKBONE_STRIDE \
        --model_dir=$MODEL_DIR \
		--kernel_size=$KERNEL_SIZE \
        --gpu_id=$2
}

function predict
{
	python pys/write_prediction.py \
        --batch_size=$EVAL_BATCH \
        --image_size $IMAGE_SIZE $IMAGE_SIZE \
        --reg=$REG \
        --data=$1 \
		--aspp_rates$RATES \
		--keep_prob=$KEEP_PROB \
		--backbone_stride=$BACKBONE_STRIDE \
        --model_dir=$MODEL_DIR \
		--kernel_size=$KERNEL_SIZE \
        --gpu_id=$2 \
		--mask_dir=$3
}

function eval_once
{
    python evaluate.py \
        --batch_size=$EVAL_BATCH \
        --image_size $IMAGE_SIZE $IMAGE_SIZE \
        --reg=$REG \
        --data=$1 \
		--aspp_rates$RATES \
		--kernel_size=$KERNEL_SIZE \
        --model_dir=$MODEL_DIR \
		--backbone_stride=$BACKBONE_STRIDE \
		--keep_prob=$KEEP_PROB \
        --gpu_id=$2
}

function eval_offline
{
	eval_func="cnn_output"
	if [ -n "$3" ]; then
		eval_func="$3"
	fi
    python pys/eval_with_crf.py \
        --batch_size=$EVAL_BATCH \
        --image_size $IMAGE_SIZE $IMAGE_SIZE \
        --data=$1 \
		--aspp_rates$RATES \
		--kernel_size=$KERNEL_SIZE \
        --model_dir=$MODEL_DIR \
		--backbone_stride=$BACKBONE_STRIDE \
		--keep_prob=$KEEP_PROB \
		--eval_func=$eval_func \
        --gpu_id=$2 
}


############################################

###############
#  Setting    #
###############
BACKBONE="resnet_v1_101"
RESNET_CKPT="resnet_v1_101.ckpt"
IMAGE_SIZE=384
REG=1e-4
MODEL_DIR="/tmp/fcn-4"
# ----------------------------------------------> GPU
TRAIN_GPU_ID=0
EVAL_GPU_ID=1
SOLVER="adam"
BACKBONE_STRIDE=8
KEEP_PROB=1.0
KERNEL_SIZE=3
RATES=" 12"

TRAIN_BATCH=4
EVAL_BATCH=8
#TRAIN_DATA="/tmp/voc2012_segpred_trainaug.tfrecord"
#TRAIN_DATA="/tmp/voc2012_segpred_trainaug.tfrecord"
TRAIN_DATA="/tmp/voc2012_seg_trainaug.tfrecord"
EVAL_DATA="/tmp/voc2012_seg_val.tfrecord"
PREDICT_DATA="/tmp/voc2012_segpred_trainaug.tfrecord"


##################
# Running        #
##################
cd ..
trap "exit" INT


if [ "$1" == "train" ]; then
    echo "In training"
	SOLVER="momentum"
	#pre_run_op
    LR=1e-3
	LRP=0.1
    train_n_epochs 20

elif [ "$1" == "eval_online" ]; then
	echo "*******************************************************"
	echo "*             In evaluation(online)                   *"
	echo "*******************************************************"
    #CUDA_VISIBLE_DEVICES=1 eval_online voc2012-val-SegMain $EVAL_GPU_ID
    CUDA_VISIBLE_DEVICES=1 eval_online $EVAL_DATA $EVAL_GPU_ID
elif [ "$1" == "eval_offline" ]; then
	echo "*******************************************************"
	echo "*             In evaluation(offline)                  *"
	echo "*******************************************************"
    eval_offline $EVAL_DATA $EVAL_GPU_ID $2
elif [ "$1" == "predict" ]; then
	echo "*******************************************************"
	echo "*                     In prediction                   *"
	echo "*******************************************************"
	predict $PREDICT_DATA 0 $2

else
    echo "Parameter '$1' not allowed"
fi
