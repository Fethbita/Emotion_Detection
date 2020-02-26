#!/bin/sh

if [ ! -d stanford-corenlp-full-2018-10-05 ]; then
    wget https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
    unzip -q stanford-corenlp-full-2018-10-05.zip
    rm stanford-corenlp-full-2018-10-05.zip
fi

cd stanford-corenlp-full-2018-10-05
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit  -port 9080 -timeout 15000

