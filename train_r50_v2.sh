#!/usr/bin/env bash
CUDA='0,1,2,3,4,5,6,7'
BATCH_SIZE='512'

TAG=r50_glintasia_v2
DATA_DIR=/home/meiyiyang/Face/insightface/datasets/faces_glint
MODEL_OUTPUT_DIR='/data/meiyi/models/r50/model_'$TAG'/model_'$TAG

LAST_EPOCH=0
PRETRAINED='/data/meiyi/models/r50_old/model_r50_v1_4_ms_0129/model-r50,81'

NETWORK=r50
FINE_TUNE=2

TARGET=classify_suning_v2,lfw

#export MXNET_CPU_WORKER_NTHREADS=2;
#export MXNET_GPU_WORKER_NTHREADS=2;
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1;
#export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice;
export CUDA_VISIBLE_DEVICES=$CUDA;
python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 5 --margin-m 0.3 --per-batch-size $BATCH_SIZE  --pretrained $PRETRAINED --verbose 500 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --fine_tune $FINE_TUNE
