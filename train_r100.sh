CUDA='0,1,2,3,4,5'
BATCH_SIZE='64'

DATA_DIR=~/Face/insightface/datasets/faces_emore
MODEL_OUTPUT_DIR=/data/meiyi/models/r100/model_r100_emore_v3_1/model_r100_emore_v3
TAG=r100_emore_v3

LAST_EPOCH=0
PRETRAIN=/data/meiyi/models/r100/model_r100_emore_v2_2/model_r100_emore_v2,20

NETWORK=r100

TARGET=lfw,classify_megaface,classify_suning_test
#LOGFILE='/home/Face/insightface/output/log/log_'$TAG'_arg.txt'

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 5 --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --last_epoch $LAST_EPOCH --pretrained $PRETRAIN
