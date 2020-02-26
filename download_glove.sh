#!/bin/sh

if [ ! -f Pickle/gloveModel ]; then
    wget https://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip
    rm glove.840B.300d.zip
    ./create_glove_model.py -f glove.840B.300d.txt
    rm glove.840B.300d.txt

    mkdir -p Pickle
    mv glove.840B.300d.txt.bin Pickle/gloveModel
    mkdir -p models
fi

