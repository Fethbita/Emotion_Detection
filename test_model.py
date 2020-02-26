#!/bin/python

import os
import sys
import time
import _pickle as cPickle
import torch
import numpy as np
from nltk.parse.corenlp import CoreNLPParser
from LSTM_model import LSTM

device = torch.device("cuda:0")
dtype = torch.float


def read_glove_vectors(glove_model_file):
    # @title Read GloVe Vectors
    print("Reading Glove Model")
    start_time = time.time()
    with open(glove_model_file, 'rb') as infile:
        glove_model = cPickle.load(infile)
    print("--- %s seconds ---" % (time.time() - start_time))
    return glove_model


def get_input_vector(glove_model, sentence):
    sentence_length = len(sentence)
    padded_vector = np.zeros((sentence_length, 1, 300))
    sentence_length = torch.tensor(
        [sentence_length]).to(device, dtype=torch.long)
    j = 0
    for word in sentence:
        if word in glove_model:
            padded_vector[j, 0] = glove_model[word]
        else:
            padded_vector[j, 0] = np.zeros(300)
        j += 1
    padded_vector = torch.from_numpy(padded_vector).to(device, dtype=dtype)
    return padded_vector, sentence_length


def start_testing(trained_model_file):
    parser = CoreNLPParser(url='http://localhost:9080')

    emotions = ['happiness', 'sadness', 'anger', 'disgust', 'surprise', 'fear']

    dirname = os.path.dirname(os.path.realpath(__file__)) + "/"

    glove_model = read_glove_vectors(dirname + "Pickle/gloveModel")

    hidden_size = 512
    num_layers = 1
    bidirectional = False
    batchnorm = False
    dropout_hidden = 0
    dropout_output = 0.9
    model = LSTM(300, hidden_size, num_layers, bidirectional,
                 batchnorm, dropout_hidden, dropout_output).to(device)

    with torch.no_grad():
        model.load_state_dict(torch.load(trained_model_file))
        print(model)
        model.eval()
        while True:
            test_sentence = input("Give a test sentence: ")
            sentence = list(parser.tokenize(test_sentence))
            input1, sent_length = get_input_vector(glove_model, sentence)
            class_pred = model(input1, sent_length)
            print("Sentence: " + test_sentence)
            _, pred = class_pred.max(dim=1)
            print("Prediction:\t" + emotions[pred[0]])
            print("Output Values:")
            percentages = torch.nn.functional.softmax(class_pred, dim=1) * 100
            for i in range(len(emotions)):
                print(emotions[i] + " %" +
                      str(percentages.data.tolist()[0][i]))


def usage():
    print("Usage:")
    print("<trained_model_file.pth>")
    sys.exit(1)


if len(sys.argv) != 2:
    usage()
else:
    start_testing(sys.argv[1])
