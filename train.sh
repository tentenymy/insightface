export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

CUDA='3'
DATA_DIR=/home/meiyi/Face/insighrface_v2/datasets/face_emore
PREFIX=/disk1/meiyi/models/insightface_mobile/model_mb1_em_v0_0
TAG=mb1_em_v0
NETWORK=y1
TARGET=lfw,classify_megaface

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_arcface_v4.py --network $NETWORK --loss-type 5 --margin-m 0.5 --per-batch-size 64 --verbose 1000 --tag $TAG --target "" --data-dir $DATA_DIR  --prefix $PREFIX