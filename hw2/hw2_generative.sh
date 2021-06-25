#!/bin/bash

# time python3 train.py --train_x=$1 --train_y=$2 --method=generative --degree=30
time python3 test.py --test_x=$1 --output_file=$2 --method=generative --degree=30