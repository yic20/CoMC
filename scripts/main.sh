#!/bin/bash

cd ..

# custom config
DATA=/disk1/lyc/datasets
TRAINER=comc

DATASET=$1
CFG=$2  # config file
run_ID=$3

export CUDA_VISIBLE_DEVICES=1

for SEED in 1 2 3
do
    DIR=output/${run_ID}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job andsave the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} 
    fi
done

# bash main.sh coco2014 rn50_coco2014 comc_coco
