#!/bin/bash

time python3 train.py --root=$1 --max_length=50 --batch_size=64 --learning_rate=0.0001 --epoch=40 --beam_size=4