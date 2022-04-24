#!/bin/bash

NVIEW=288

python train_denoising.py "./config/l2_depth_4.cfg" --Train.device "'1'" \
--IO.x_dir "'mayo/data/${NVIEW}'" --IO.tag "'${NVIEW}/l2_depth_4'" \
&> ./log/l2_depth_4_${NVIEW}.log

python train_denoising.py "./config/l2_depth_4_large.cfg" --Train.device "'1'" \
--IO.x_dir "'mayo/data/${NVIEW}'" --IO.tag "'${NVIEW}/l2_depth_4_large'" \
&> ./log/l2_depth_4_large_${NVIEW}.log