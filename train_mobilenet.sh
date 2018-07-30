CUDA='1,2,3'
BATCH_SIZE='152'

DATA_DIR=/home/meiyi/Face/insightface_v2/datasets/faces_emore
MODEL_OUTPUT_DIR=/disk1/meiyi/models/insightface_mobile/model_mobile1_emore_v0_0/model_mobile1_emore_v0
TAG=mobile1_emore_v0

LAST_EPOCH=166
PRETRAIN=/disk1/meiyi/models/insightface_mobile/model_mobile1_emore_v0_0/model_mobile1_emore_v0,166

NETWORK=m1
EMB_SIZE='128'
FC7_WD_MULT='1.0'
MOM='0.9'

TARGET=lfw,classify_megaface
LOGFILE='/home/meiyi/Face/insightface/output/log/log_'$TAG'_arg.txt'

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 5 --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --emb-size $EMB_SIZE --fc7-wd-mult $FC7_WD_MULT --mom $MOM --last_epoch $LAST_EPOCH --pretrained $PRETRAIN
