#!/bin/bash

time python3 -W ignore Word2Vec_train.py --train_data=$1 --nolabel_data=$2 --test_data=$3 --embedding_dimension=256 --iteration=10
time python3 -W ignore RNN_train.py --train_data=$1 --nolabel_data=$2 --max_length=32 --batch_size=32 --learning_rate=0.001 --epoch=5 --threshold_1=0.1 --threshold_2=0.9