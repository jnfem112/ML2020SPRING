#!/bin/bash

gdown --id '1B8ljdrxYXJsZv2vmTequdPOofp3VF3NN' --output teacher_resnet18.bin
time python3 train.py --train_directory=$1 --validation_directory=$2 --batch_size=32 --learning_rate=0.002 --epoch=150