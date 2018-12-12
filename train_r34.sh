#!/usr/bin/env bash
CUDA='4,5,6,7'
BATCH_SIZE='256'

TAG=r34_glintasia_v0
DATA_DIR=/home/meiyiyang/Face/insightface/datasets/faces_glint
MODEL_OUTPUT_DIR='/data/meiyi/models/r34/model_'$TAG'/model_'$TAG

LAST_EPOCH=0
PRETRAINED='/data/meiyi/models/r34/original/model,0'

NETWORK=r34
FINE_TUNE=2

TARGET=classify_suning_v2,lfw

export MXNET_CPU_WORKER_NTHREADS=2;
export MXNET_GPU_WORKER_NTHREADS=2;
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1;
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice;
export CUDA_VISIBLE_DEVICES=$CUDA;
python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 5 --margin-m 0.3 --per-batch-size $BATCH_SIZE  --pretrained $PRETRAINED --verbose 500 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --fine_tune $FINE_TUNE
