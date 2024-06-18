#!/bin/bash

cd ..

# custom config
DATA=/disk1/lyc/datasets
TRAINER=comc

DATASET=$1
CFG=$2  # config file
run_ID=$3

export CUDA_VISIBLE_DEVICES=1

for SEED in 1
do
    DIR=experiment/test/${run_ID}/seed${SEED}
    echo "Run this job andsave the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR}\
    --model-dir output/${run_ID}/${TRAINER}/${CFG}/seed${SEED}\
    --eval-only
    # fi
done
