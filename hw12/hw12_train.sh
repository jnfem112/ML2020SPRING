#!/bin/bash

time python3 train.py --directory=$1 --batch_size=128 --learning_rate=0.00002 --weight_decay=0.0005 --epoch=2000