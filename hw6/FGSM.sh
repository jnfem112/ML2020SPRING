#!/bin/bash

time python3 FGSM.py --input_directory=$1 --output_directory=$2 --epsilon=0.01
time python3 judge.py --input_directory=$1 --output_directory=$2
