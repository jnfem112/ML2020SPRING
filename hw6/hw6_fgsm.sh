#!/bin/bash

time python3 attack.py --input_directory=$1 --output_directory=$2 --method='FGSM' --epsilon=0.01
# time python3 judge.py --input_directory=$1 --output_directory=$2