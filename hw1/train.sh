#!/bin/bash

time python3 train.py --train_data=$1 --batch_size=1024 --learning_rate=0.001 --epoch=2000
