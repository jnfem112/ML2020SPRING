#!/bin/bash

time python3 RNN_test.py -W ignore --test_data=$1 --output_file=$2 --max_length=32 --batch_size=1024
