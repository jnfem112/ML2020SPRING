#!/bin/bash

wget https://www.dropbox.com/s/d7t9lqnastp98x2/CNN.pkl?dl=0 -O CNN.pkl
python3 CNN_test.py --test_directory=$1 --output_file=$2 --batch_size=1024
