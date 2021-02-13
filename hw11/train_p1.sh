#!/bin/bash

time python3 train.py --train_directory=$1 --checkpoint=$2 --model=DCGAN --input_dim=100 --batch_size=64 --learning_rate=0.0001 --epoch=10