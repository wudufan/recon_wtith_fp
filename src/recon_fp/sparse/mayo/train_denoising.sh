#!/bin/bash

python train_denoising.py "./config/l2_depth_4.cfg" --Train.device "'2'" \
--IO.x_dir "'mayo/data/144'" --IO.tag "'144/l2_depth_4'" \
&> ./log/l2_depth_4_144.log &

python train_denoising.py "./config/l2_depth_4.cfg" --Train.device "'3'" \
--IO.x_dir "'mayo/data/96'" --IO.tag "'96/l2_depth_4'" \
&> ./log/l2_depth_4_96.log &

wait

python train_denoising.py "./config/l2_depth_4_large.cfg" --Train.device "'2'" \
--IO.x_dir "'mayo/data/144'" --IO.tag "'144/l2_depth_4_large'" \
&> ./log/l2_depth_4_large_144.log &

python train_denoising.py "./config/l2_depth_4_large.cfg" --Train.device "'3'" \
--IO.x_dir "'mayo/data/96'" --IO.tag "'96/l2_depth_4_large'" \
&> ./log/l2_depth_4_large_96.log &

wait