#!/bin/bash

set -e
set -x

mkdir -p build
cd src
LUCENE_PATH=../../../tools/lucene-6.0.0
CP_PATH=$LUCENE_PATH/core/lucene-core-6.0.0.jar:$LUCENE_PATH/queryparser/lucene-queryparser-6.0.0.jar:$LUCENE_PATH/analysis/common/lucene-analyzers-common-6.0.0.jar:$LUCENE_PATH/demo/lucene-demo-6.0.0.jar:. 
javac -cp $CP_PATH -d ../build antnlp/opie/indexsearch/*.java 

cd ../build
jar cvf ../opie-indexsearch.jar antnlp/opie/indexsearch/*.class

cd ..



