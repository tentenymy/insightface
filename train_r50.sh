#!/usr/bin/env bash
CUDA='6,7'
BATCH_SIZE='256'

TAG=r50_asia_v0
DATA_DIR=/data/meiyi/glint/faces_asia
MODEL_OUTPUT_DIR='/data/meiyi/models/r50/model_'$TAG'/model_'$TAG

LAST_EPOCH=0
PRETRAINED='/data/meiyi/models/r50/model_r50_emore_v0/model_r50_emore_v0,19'

NETWORK=r50
FINE_TUNE=2

TARGET=classify_suning_v2,classify_megaface,classify_megaface_fgnet

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 5 --margin-m 0.3 --per-batch-size $BATCH_SIZE  --pretrained $PRETRAINED --verbose 500 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --fine_tune $FINE_TUNE
