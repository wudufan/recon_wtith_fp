#!/bin/bash

NVIEWS=(288 144 96)
NAMES=("ACR" "L291" "L067" "L143")
DEVICE="3"
for NVIEW in "${NVIEWS[@]}"
do
    for NAME in "${NAMES[@]}"
    do
        python postprocess_with_fp.py --device $DEVICE \
        --name $NAME --nview $NVIEW \
        --input_dir "mayo/train/${NVIEW}/l2_depth_4/valid/" \
        --prj_dir "mayo/data/${NVIEW}/" \
        --output_dir "mayo/train/${NVIEW}/l2_depth_4/prj_recon/" \
        &> log/postprocess_with_fp/${NAME}_${NVIEW}.log
    done
done