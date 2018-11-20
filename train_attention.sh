CUDA='1,2,3,4,5,6,7'
BATCH_SIZE='20'

DATA_DIR=/home/meiyiyang/Face/insightface/datasets/faces_emore
TAG=attention92_emore_v0
MODEL_OUTPUT_DIR='/data/meiyi/models/r100/model_'$TAG'/model_'$TAG

LAST_EPOCH=0
PRETRAIN='/data/meiyi/models/r100/model_'$TAG'/model_'$TAG',449'

NETWORK='a92'

TARGET=lfw,classify_megaface

export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64;
export CUDA_VISIBLE_DEVICES=$CUDA;
python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 4 --per-batch-size $BATCH_SIZE --verbose 1000 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --last_epoch $LAST_EPOCH
