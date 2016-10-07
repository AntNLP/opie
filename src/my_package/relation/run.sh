#!/bin/bash
# python lstm_cnn_train.py -d reviews_Cell_Phones_and_Accessories --num_filters=128 --n_hidden=2 --drop_prob=0.5 --l2=0 --batch_size=64 --num_epochs=1 --complex_op=1
python lstm_cnn_train.py -d reviews_Home_and_Kitchen --num_filters=128 --n_hidden=2 --drop_prob=0.5 --l2=0 --batch_size=64 --num_epochs=1 --complex_op=1

