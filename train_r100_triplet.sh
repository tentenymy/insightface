CUDA='0,1,2,3,4,5'
BATCH_SIZE=75

DATA_DIR=/home/meiyiyang/Face/insightface/datasets/faces_emore
MODEL_OUTPUT_DIR=/data/meiyi/models/r100/model_r100_emore_v3_1/model_r100_emore_v3
TAG=r100_emore_v3

LAST_EPOCH=12
PRETRAIN=/data/meiyi/models/r100/model_r100_emore_v3_1/model_r100_emore_v3,12

NETWORK=r100
MOM='0.0'
LOSS_TYPE=12
TRIPLET_BAG_SIZE=2700

TARGET=classify_megaface,lfw,classify_suning_test
LOGFILE='/home/meiyiyang/Face/insightface/output/log/log_'$TAG'_arg.txt'

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train.py --network $NETWORK --loss-type $LOSS_TYPE --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --mom $MOM --pretrained $PRETRAIN --lr 0.005 --tag $TAG --last_epoch $LAST_EPOCH --triplet-bag-size $TRIPLET_BAG_SIZE
