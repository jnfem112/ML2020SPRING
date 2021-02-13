#!/bin/bash

time python3 -W ignore train.py --train_x=$1 --checkpoint=$2 --data_augmentation=1 --batch_size=16 --learning_rate=0.0001 --weight_decay=0.00001 --epoch=100