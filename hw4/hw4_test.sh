#!/bin/bash

wget https://www.dropbox.com/s/d7k4ohp0r49byu4/RNN.pkl?dl=0 -O RNN.pkl
wget https://www.dropbox.com/s/kit9oza71798jdo/Word2Vec.model?dl=0 -O Word2Vec.model
wget https://www.dropbox.com/s/cm24pugga75hbzm/Word2Vec.model.wv.vectors.npy?dl=0 -O Word2Vec.model.wv.vectors.npy
wget https://www.dropbox.com/s/1aeyl9paykmhueq/Word2Vec.model.trainables.syn1neg.npy?dl=0 -O Word2Vec.model.trainables.syn1neg.npy
time python3 -W ignore RNN_test.py --test_data=$1 --output_file=$2 --max_length=32