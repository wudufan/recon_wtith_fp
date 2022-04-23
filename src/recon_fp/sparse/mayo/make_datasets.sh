#!/bin/bash

python make_datasets.py --nview 2304
python make_datasets.py --nview 144 --save_prj 1
python make_datasets.py --nview 96 --save_prj 1