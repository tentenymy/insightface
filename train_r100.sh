#!/usr/bin/env bash
CUDA='4,5,6,7'
BATCH_SIZE='256'

TAG=r100_glint_v0
DATA_DIR=/data/meiyi/glint/faces_glintasia
MODEL_OUTPUT_DIR='/data/meiyi/models/r100/model_'$TAG'/model_'$TAG

LAST_EPOCH=0
PRETRAINED='/data/meiyi/models/r100/model_r100_emore_v4_3_tri/model_r100_emore_v4,12'

NETWORK=r100

TARGET=classify_megaface,classify_megaface_fgnet

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 5 --margin-m 0.3 --per-batch-size $BATCH_SIZE  --pretrained $PRETRAINED --verbose 1000 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --fine_tune 1
