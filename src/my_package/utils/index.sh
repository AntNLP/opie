#!/bin/bash

set -e
set -x

if test $1=''
then
    DOMAIN=reviews_Cell_Phones_and_Accessories
else
    DOMAIN=$1
fi
echo "DOMAIN NAME: "$DOMAIN
LUCENE_PATH=$OPIE_DIR/tools/lucene-6.0.0
DOCS_PATH=$OPIE_DIR/data/domains/$DOMAIN/docs/doc
INDEX_PATH=$OPIE_DIR/data/domains/$DOMAIN/index
CP_PATH=$LUCENE_PATH/core/lucene-core-6.0.0.jar:$LUCENE_PATH/queryparser/lucene-queryparser-6.0.0.jar:$LUCENE_PATH/analysis/common/lucene-analyzers-common-6.0.0.jar:$LUCENE_PATH/demo/lucene-demo-6.0.0.jar:$OPIE_DIR/src/indexsearch/opie-indexsearch.jar

java -cp $CP_PATH antnlp.opie.indexsearch.IndexFiles -docs $DOCS_PATH -index $INDEX_PATH > /tmp/index.log
