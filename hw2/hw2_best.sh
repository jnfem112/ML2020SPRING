#!/bin/bash

# time python3 train.py --train_x=$1 --train_y=$2 --method=logistic --degree=10 --learning_rate=0.05 --lambd=0.001 --epoch=1000
time python3 test.py --test_x=$1 --output_file=$2 --method=logistic --degree=10