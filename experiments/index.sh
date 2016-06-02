#!/bin/bash

set -e
set -x


LUCENE_PATH=../tools/lucene-6.0.0
DOCS_PATH=../data/domains/reviews_Cell_Phones_and_Accessories/docs
INDEX_PATH=../data/domains/reviews_Cell_Phones_and_Accessories/index
CP_PATH=$LUCENE_PATH/core/lucene-core-6.0.0.jar:$LUCENE_PATH/queryparser/lucene-queryparser-6.0.0.jar:$LUCENE_PATH/analysis/common/lucene-analyzers-common-6.0.0.jar:$LUCENE_PATH/demo/lucene-demo-6.0.0.jar:../src/indexsearch/opie-indexsearch.jar

java -cp $CP_PATH antnlp.opie.indexsearch.IndexFiles -docs $DOCS_PATH -index $INDEX_PATH
