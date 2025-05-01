#!/bin/bash
python tda_runner.py --config configs --datasets I --backbone RN50 --attack pgd --epsilon 0.03 --alpha 0.007 --iters 10