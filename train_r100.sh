CUDA='0,1,2,3'
BATCH_SIZE='100'

DATA_DIR=~/Face/insightface/datasets/faces_emore
MODEL_OUTPUT_DIR=/data/meiyi/models/r100/model_r100_emore_v0_0/model_r100_emore_v0
TAG=r100_emore_v0

LAST_EPOCH=0
PRETRAIN=/data/meiyi/models/r100/model_r100_v0/model_r100_v0,450

NETWORK=r100
EMB_SIZE='512'
FC7_WD_MULT='1.0'
MOM='0.0'

TARGET=lfw,classify_megaface
#LOGFILE='/home/Face/insightface/output/log/log_'$TAG'_arg.txt'

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 5 --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --emb-size $EMB_SIZE --fc7-wd-mult $FC7_WD_MULT --mom $MOM --last_epoch $LAST_EPOCH --pretrained $PRETRAIN
