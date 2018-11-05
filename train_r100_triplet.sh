CUDA='2,3,4,5,6,7'
BATCH_SIZE=75

DATA_DIR=/home/meiyiyang/Face/insightface/datasets/faces_emore
TAG=r100_emore_v4
MODEL_OUTPUT_DIR='/data/meiyi/models/r100/model_'$TAG'_3_tri/model_'$TAG

LAST_EPOCH=0
PRETRAIN=/data/meiyi/models/r100/model_r100_emore_v4_0/model_r100_emore_v4,134

NETWORK=r100
MOM='0.0'
LOSS_TYPE=12
TRIPLET_BAG_SIZE=9000
TRIPLET_ALPHA=0.4

TARGET=classify_megaface
LOGFILE='/home/meiyiyang/Face/insightface/output/log/log_'$TAG'_arg.txt'

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train.py --network $NETWORK --loss-type $LOSS_TYPE --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --mom $MOM --pretrained $PRETRAIN --lr 0.005 --tag $TAG --last_epoch $LAST_EPOCH --triplet-bag-size $TRIPLET_BAG_SIZE --triplet-alpha $TRIPLET_ALPHA
