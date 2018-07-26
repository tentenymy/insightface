export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

CUDA='3'
DATA_DIR=/home/meiyi/Face/insightface_v2/datasets/faces_emore
PREFIX=/disk1/meiyi/models/insightface_mobile/model_moible1_emore_v0_0
TAG=mobile1_emore_v0
NETWORK=y1
TARGET=lfw,suningshop_v9,lassify_megaface

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 5 --margin-m 0.5 --per-batch-size 64 --verbose 1000 --tag $TAG --target "" --data-dir $DATA_DIR  --prefix $PREFIX
