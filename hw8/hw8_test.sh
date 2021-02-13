#!/bin/bash

wget https://www.dropbox.com/s/49h0q449rkn5px3/Seq2Seq.pkl?dl=0 -O Seq2Seq.pkl
time python3 test.py --root=$1 --output_file=$2 --max_length=50 --beam_size=4