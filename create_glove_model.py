#!/usr/bin/env python3
import sys
import time
import pickle
import numpy as np


def load_and_pickle(filename):
    start_time = time.time()
    glove_model = load_glove_model(filename)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    pickle_model(filename, glove_model)
    print("--- %s seconds ---" % (time.time() - start_time))


def load_glove_model(filename):
    print("Loading Glove Model")
    model = {}
    with open(filename, 'r', encoding="utf8") as f:
        for line in f:
            split_line = line.split(' ')
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    print("Done.", len(model), "words loaded!")
    return model


def pickle_model(filename, glove_model):
    print("Writing Glove Model Binary")
    with open(filename + ".bin", 'wb') as outfile:
        pickle.dump(glove_model, outfile)


def usage():
    print("Usage:")
    print("-f <filename>")
    sys.exit(1)


if len(sys.argv) != 3:
    usage()
elif sys.argv[1] == '-f':
    load_and_pickle(sys.argv[2])
else:
    usage()
