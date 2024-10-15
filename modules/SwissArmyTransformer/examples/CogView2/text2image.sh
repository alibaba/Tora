#!/bin/bash

SAT_HOME=/sharefs/cogview-new python cogview2_text2image.py \
       --mode inference \
       --fp16 \
       --input-source input.txt \
       --output-path samples_sat_v0.2 \
       --batch-size 16 \
       --max-inference-batch-size 8 \
       $@

