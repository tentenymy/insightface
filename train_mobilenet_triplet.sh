CUDA='0,1,2,3,4,5'
BATCH_SIZE='144'

DATA_DIR=/home/meiyiyang/Face/insightface/datasets/faces_emore
MODEL_OUTPUT_DIR=/data/meiyi/models/mobile/model_mobile1_emore_v0_1tri/model_mobile1_emore_v0
TAG=mobile1_emore_v0

LAST_EPOCH=242
PRETRAIN=/data/meiyi/models/mobile/model_mobile1_emore_v0_0/model_mobile1_emore_v0,242

NETWORK=m1
EMB_SIZE='128'
FC7_WD_MULT='1.0'
MOM='0.0'
LOSS_TYPE=12

TARGET=lfw,classify_megaface
LOGFILE='/home/meiyiyang/Face/insightface/output/log/log_'$TAG'_arg.txt'

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type $LOSS_TYPE --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --emb-size $EMB_SIZE --fc7-wd-mult $FC7_WD_MULT --mom $MOM --last_epoch $LAST_EPOCH --pretrained $PRETRAIN
