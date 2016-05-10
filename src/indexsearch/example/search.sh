#!/bin/bash

set -e
set -x


LUCENE_PATH=../../../tools/lucene-6.0.0
CP_PATH=$LUCENE_PATH/core/lucene-core-6.0.0.jar:$LUCENE_PATH/queryparser/lucene-queryparser-6.0.0.jar:$LUCENE_PATH/analysis/common/lucene-analyzers-common-6.0.0.jar:$LUCENE_PATH/demo/lucene-demo-6.0.0.jar:../opie-indexsearch.jar

java -cp $CP_PATH antnlp.opie.indexsearch.SearchFiles -index index -queries tests/queries
