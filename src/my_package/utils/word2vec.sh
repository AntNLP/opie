#!/bin/bash

if test $1=''
then
    DOMAIN=reviews_Cell_Phones_and_Accessories
else
    DOMAIN=$1
fi
TRAIN_PATH=$OPIE_DIR/data/domains/$DOMAIN/multiopinexpr/clean_replace_text
EVAL_PATH=$OPIE_DIR/data/domains/$DOMAIN/multiopinexpr/word2vec/questions-words.txt
SAVE_PATH=$OPIE_DIR/data/domains/$DOMAIN/multiopinexpr/word2vec/
python $OPIE_DIR/src/my_package/utils/word2vec_optimized.py --train_data $TRAIN_PATH --eval_data $EVAL_PATH --save_path $SAVE_PATH
