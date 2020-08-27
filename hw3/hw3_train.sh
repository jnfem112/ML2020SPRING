#!/bin/bash

time python3 train.py --train_directory=$1 --validation_directory=$2 --batch_size=128 --learning_rate=0.001 --epoch=150