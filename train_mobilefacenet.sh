# environment
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

# gpu & batch
CUDA='1,3'
BATCH_SIZE='192'

# training dataset
DATA_DIR=/home/meiyi/Face/insightface_v2/datasets/faces_emore

# model
MODEL_OUTPUT_DIR=/disk1/meiyi/models/insightface_mobile/model_moible1_emore_v0_0
TAG=mobile1_emore_v0
NETWORK=y1
EMB_SIZE='128'
FC7_WD_MULT='10.0'
MOM='0.9'

# testing dataset
TARGET=lfw,suningshop_v9,classify_megaface

export CUDA_VISIBLE_DEVICES=$CUDA; python -u ~/Face/insightface/src/train_softmax_my.py --network $NETWORK --loss-type 5 --margin-m 0.5 --per-batch-size $BATCH_SIZE --verbose 1000 --tag $TAG --target $TARGET --data-dir $DATA_DIR  --prefix $MODEL_OUTPUT_DIR --emb-size $EMB_SIZE --fc7-wd-mult $FC7_WD_MULT --mom $MOM
