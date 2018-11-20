CUDA='2,3,4,5,6,7'
BATCH_SIZE='200'

DATA_DIR=/home/meiyiyang/Face/insightface/datasets/faces_emore
TAG=mnasnet_emore_v1
MODEL_OUTPUT_DIR='/data/meiyi/models/mobile/model_'$TAG'/model_'$TAG

LAST_EPOCH=0
PRETRAIN=''

NETWORK=m3

TARGET=lfw,classify_megaface

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64;
export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 4 --per-batch-size $BATCH_SIZE --verbose 1000 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --last_epoch $LAST_EPOCH
