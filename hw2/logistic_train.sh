#!/bin/bash

time python3 logistic_train.py --train_x=$1 --train_y=$2 --lambd=0.001 --learning_rate=0.05 --epoch=1000
