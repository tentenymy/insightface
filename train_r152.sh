CUDA='2,3,4,5,6,7'
BATCH_SIZE='16'

TAG=r152_emore_v0
DATA_DIR=~/Face/insightface/datasets/faces_emore
MODEL_OUTPUT_DIR='/data/meiyi/models/r100/model_'$TAG'_0/model_'$TAG

LAST_EPOCH=0
PRETRAIN='/data/meiyi/models/r100/model_r100_emore_v4_0/model_r100_emore_v4,86'

NETWORK=r152

TARGET=classify_megaface
#LOGFILE='/home/Face/insightface/output/log/log_'$TAG'_arg.txt'

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 4 --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --last_epoch $LAST_EPOCH
