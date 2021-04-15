#!/usr/bin/env python3

import sys
import os
import _pickle as cPickle
from nltk.parse.corenlp import CoreNLPParser


def create_dataset_bin(annotation_file, data_file):
    parser = CoreNLPParser(url='http://localhost:9080')
    dirname = os.path.dirname(os.path.realpath(__file__)) + "/"

    dataset = []

    with open(annotation_file, "r") as file1, open(data_file, "r") as file2:
        for line_from_file_1, line_from_file_2 in zip(file1, file2):
            output = None
            line1 = line_from_file_1.split()
            line2 = line_from_file_2
            if line1[0] == "ne":
                output = 7
            elif line1[0] == "hp":
                output = 0
            elif line1[0] == "sd":
                output = 1
            elif line1[0] == "ag":
                output = 2
            elif line1[0] == "dg":
                output = 3
            elif line1[0] == "sp":
                output = 4
            elif line1[0] == "fr":
                output = 5
            elif line1[0] == "me":
                output = 6
            dataset.append((output, list(parser.tokenize(line2))))
    print(len(dataset))

    with open(dirname + "Pickle/dataset_ready", 'wb') as outfile:
        cPickle.dump(dataset, outfile)


def usage():
    print("Usage:")
    print("<annotation filename> <data filename>")
    sys.exit(1)


if len(sys.argv) != 3:
    usage()
else:
    create_dataset_bin(sys.argv[1], sys.argv[2])
