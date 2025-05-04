#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner.py     --config configs \
                                                --datasets I \
                                                --backbone RN50 \
                                                --attack fgsm \
                                                --epsilon 0.03