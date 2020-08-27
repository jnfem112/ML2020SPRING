#!/bin/bash

wget https://www.dropbox.com/s/d7t9lqnastp98x2/CNN.pkl?dl=0 -O CNN.pkl
time python3 test.py --test_directory=$1 --output_file=$2