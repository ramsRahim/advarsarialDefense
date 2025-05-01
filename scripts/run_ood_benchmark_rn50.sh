#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config configs \
                                                --datasets A/V/R/S  \
                                                --attack none \
                                                --backbone RN50