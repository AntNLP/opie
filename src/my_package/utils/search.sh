#!/bin/bash

# set -e
# set -x


if test $1=''
then
    DOMAIN=reviews_Cell_Phones_and_Accessories
else
    DOMAIN=$1
fi
LUCENE_PATH=$OPIE_DIR/tools/lucene-6.0.0
INDEX_PATH=$OPIE_DIR/data/domains/$DOMAIN/index
DOCS_PATH=$OPIE_DIR/data/domains/$DOMAIN/docs/doc
CP_PATH=$LUCENE_PATH/core/lucene-core-6.0.0.jar:$LUCENE_PATH/queryparser/lucene-queryparser-6.0.0.jar:$LUCENE_PATH/analysis/common/lucene-analyzers-common-6.0.0.jar:$LUCENE_PATH/demo/lucene-demo-6.0.0.jar:$OPIE_DIR/src/indexsearch/opie-indexsearch.jar
QUERIES_PATH=$OPIE_DIR/data/domains/$DOMAIN/docs/queries
SAVE_FILE=$2
SAVE_PATH=$OPIE_DIR/data/domains/$DOMAIN/docs/$SAVE_FILE

java -cp $CP_PATH antnlp.opie.indexsearch.SearchFiles -paging 1 -index $INDEX_PATH -queries $QUERIES_PATH >> $SAVE_PATH
