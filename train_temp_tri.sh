CUDA='0,1,2,3,4,5'
BATCH_SIZE='150'

DATA_DIR=/home/meiyiyang/Face/insightface/datasets/faces_emore
MODEL_OUTPUT_DIR=/data/meiyi/models/mobile/model_mobile1_emore_v1_2tri/model_mobile1_emore_v0
TAG=mobile1_emore_v0

LAST_EPOCH=242
PRETRAIN=/data/meiyi/models/mobile/model_mobile1_emore_v0_0/model_mobile1_emore_v0,242

NETWORK=m1
EMB_SIZE='128'
FC7_WD_MULT='1.0'
MOM='0.0'
LOSS_TYPE=12
TRIPLET_BAG_SIZE=3600

TARGET=lfw,classify_megaface,classify_suning_test
LOGFILE='/home/meiyiyang/Face/insightface/output/log/log_'$TAG'_arg.txt'

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train.py --network $NETWORK --loss-type $LOSS_TYPE --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --emb-size $EMB_SIZE --mom $MOM --pretrained $PRETRAIN --lr 0.005 --tag $TAG --last_epoch $LAST_EPOCH --triplet-bag-size $TRIPLET_BAG_SIZE
