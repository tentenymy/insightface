CUDA='2,3,4,5,6,7'
BATCH_SIZE=75

DATA_DIR=/home/meiyiyang/Face/insightface/datasets/faces_emore
MODEL_OUTPUT_DIR=/data/meiyi/models/r100/model_r100_emore_v3_2/model_r100_emore_v3
TAG=r100_emore_v3

LAST_EPOCH=0
PRETRAIN=/data/meiyi/models/r100/model_r100_emore_v3_0/model_r100_emore_v3,107

NETWORK=r100
MOM='0.0'
LOSS_TYPE=12
TRIPLET_BAG_SIZE=3600
TRIPLET_ALPHA=0.25

TARGET=classify_megaface,lfw,classify_suning_test
LOGFILE='/home/meiyiyang/Face/insightface/output/log/log_'$TAG'_arg.txt'

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train.py --network $NETWORK --loss-type $LOSS_TYPE --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --mom $MOM --pretrained $PRETRAIN --lr 0.005 --tag $TAG --last_epoch $LAST_EPOCH --triplet-bag-size $TRIPLET_BAG_SIZE --triplet-alpha $TRIPLET_ALPHA
