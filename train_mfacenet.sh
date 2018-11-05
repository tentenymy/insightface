CUDA='1,2'
BATCH_SIZE='224'

DATA_DIR=/home/meiyiyang/Face/insightface/datasets/faces_emore
TAG=y1_v0
MODEL_OUTPUT_DIR='/data/meiyi/models/mobile/model_'$TAG'_0/model_'$TAG

LAST_EPOCH=141
PRETRAIN='/data/meiyi/models/mobile/model_y1_v0_0/model_y1_v0,141'

NETWORK=y1
EMB_SIZE='128'
FC7_WD_MULT='1.0'
MOM='0.9'

TARGET=classify_megaface,lfw,classify_suning_test
LOGFILE='/home/meiyiyang/Face/insightface/output/log/log_'$TAG'_arg.txt'

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 5 --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --emb-size $EMB_SIZE --fc7-wd-mult $FC7_WD_MULT --mom $MOM --last_epoch $LAST_EPOCH --pretrain $PRETRAIN
