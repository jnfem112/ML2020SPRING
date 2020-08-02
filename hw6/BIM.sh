#!/bin/bash

time python3 BIM.py --input_directory=$1 --output_directory=$2 --epsilon=0.0135
time python3 judge.py --input_directory=$1 --output_directory=$2
