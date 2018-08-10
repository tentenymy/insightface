CUDA='2,3,4,5,6,7'
BATCH_SIZE='64'

DATA_DIR=~/Face/insightface/datasets/faces_emore
MODEL_OUTPUT_DIR=/data/meiyi/models/r100/model_r100_emore_v3_0/model_r100_emore_v3
TAG=r100_emore_v3

LAST_EPOCH=92
PRETRAIN=/data/meiyi/models/r100/model_r100_emore_v3_0/model_r100_emore_v3,92

NETWORK=r100

TARGET=classify_megaface,lfw
#LOGFILE='/home/Face/insightface/output/log/log_'$TAG'_arg.txt'

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 5 --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --last_epoch $LAST_EPOCH --pretrained $PRETRAIN
