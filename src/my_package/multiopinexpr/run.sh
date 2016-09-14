#/usr/bin/bash

DOMAIN=reviews_Cell_Phones_and_Accessories
LOG_PATH=$OPIE_DIR/data/domains/$DOMAIN/multiopinexpr/logs
export CUDA_VISIBLE_DEVICES=0

# python phrase_generator.py -d $DOMAIN
# python label_propagation.py -d $DOMAIN
# python data_helpers.py -d $DOMAIN
# $OPIE_DIR/src/my_package/utils/word2vec.sh $DOMAIN | tee $LOG_PATH/word2vec 2>&1
# python train.py -filter_size "3,4,5" -num_filters 128 | \
# tee $LOG_PATH/train_embeddings_initialize 2>&1
# python train.py --filter_sizes="3" --num_filters=1000 --embedding_init=True
python review_classification.py -d $DOMAIN
