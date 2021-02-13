#!/bin/bash

wget https://www.dropbox.com/s/to4owiaaghztpwy/generator.pkl?dl=0 -O generator.pkl
wget https://www.dropbox.com/s/cc8ier909c6ylyn/classifier_1.pkl?dl=0 -O classifier_1.pkl
wget https://www.dropbox.com/s/dtyybtelrkbmi2n/classifier_2.pkl?dl=0 -O classifier_2.pkl
time python3 test.py --directory=$1 --output_file=$2