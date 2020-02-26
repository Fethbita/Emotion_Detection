#!/bin/python

# @title Imports
import multiprocessing as mp
import itertools
import gc
from PIL import Image, ImageFont, ImageDraw
from io import BytesIO
import pandas as pd
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from random import shuffle
import _pickle as cPickle
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from LSTM_model import LSTM


dirname = os.path.dirname(os.path.realpath(__file__)) + "/"

dtype = torch.float
device = torch.device("cuda:0")

emotions = ['happiness', 'sadness', 'anger', 'disgust', 'surprise', 'fear']

# @title Read GloVe Vectors
start_time = time.time()
with open(dirname + "Pickle/gloveModel", 'rb') as infile:
    gloveModel = cPickle.load(infile)
print("--- %s seconds ---" % (time.time() - start_time))

# @title Read Dataset
start_time = time.time()
with open(dirname + "Pickle/dataset_ready", 'rb') as infile:
    dataset = cPickle.load(infile)
print("--- %s seconds ---" % (time.time() - start_time))
print(len(dataset))

# @title Dataset Shuffle
shuffle(dataset)

# @title Filter out not needed data points
# [536, 173, 179, 172, 115, 115, 176, 2800]
counter = [0] * 8
filtereddata = []
for data in dataset:
    counter[data[0]] += 1
    if data[0] == 7 or data[0] == 6:
        continue
    filtereddata.append(data)
print(len(filtereddata))
print(counter)

# @title Plotting Function Definitions


pd.options.display.float_format = '{:.3f}'.format

# https://stackoverflow.com/a/43647344
# SET FONT FOR MATPLOTLIB
plt.rcParams['font.family'] = 'Source Code Pro'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# SET FONT FOR PIL
font = ImageFont.truetype("SourceCodePro-Black.ttf", 25)


def fix_report(report):
    report['accuracy'] = {'f1-score': report.pop('accuracy', None)}
    return report


def report_sort(report):
    idx = emotions + ['accuracy', 'macro avg', 'weighted avg']
    return report.reindex(idx)


def report_dict_to_df(report):
    df = pd.DataFrame.from_dict(report, orient='index')
    return df


def report_to_string(report):
    df = report_sort(report_dict_to_df(report))
    return df.to_string()


def report_average(class_reports):
    pandas_reports = [report_dict_to_df(report) for report in class_reports]
    pd_df = report_sort(pd.concat(pandas_reports).groupby(level=0).mean())
    return pd_df.to_string()


# https://stackoverflow.com/a/35599851


def plot_confusion_matrix(cm, target_names, train):
    title = r'\ confusion\ matrix'
    cmap = plt.cm.YlGn
    plt.figure(figsize=[5, 4])
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(
        r"$\bf{" + (("train" + title) if train else ("test" + title)) + "}$")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.grid(False)

    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=125)
    plt.close()

    return buffer


def plottraintest(trainlosses, testlosses):
    plt.figure(figsize=[10, 5])

    minposs = testlosses.index(min(testlosses[1:]))
    plt.axvline(minposs, linestyle='--', color='r')

    plt.plot(trainlosses)
    plt.plot(testlosses)
    plt.title(r"$\bf{" + r'model\ train\ vs\ test\ loss' + "}$")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Early Stopping Checkpoint',
                'train', 'test'], loc='lower left')
    plt.ylim(0, 2)
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=125)
    plt.close()

    return buffer


def getSize(text):
    img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(img)
    return draw.textsize(text, font)


def appendHorizontal(images, space_in_between):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new(
        'RGB', (total_width + space_in_between, max_height), (56, 56, 56))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0] + space_in_between

    return new_im


def appendVertical(images, space_in_between):
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (max_width, total_height +
                               space_in_between), (56, 56, 56))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1] + space_in_between

    return new_im


def showResults(train, test, train_conf_matrix, test_conf_matrix, emotions):
    images = []
    for text in [train, test]:
        text_width, text_height = getSize(text)
        img = Image.new(
            'RGB', (text_width + 50, text_height + 50), (56, 56, 56))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), text, fill=(255, 255, 255), font=font)
        images.append(img)

    new_im = appendHorizontal(images, 100)

    train_conf_matrix = plot_confusion_matrix(
        train_conf_matrix, emotions, True)
    test_conf_matrix = plot_confusion_matrix(test_conf_matrix, emotions, False)

    train_conf_matrix.seek(0)
    test_conf_matrix.seek(0)
    train_conf_image = Image.open(train_conf_matrix)
    test_conf_image = Image.open(test_conf_matrix)
    images2 = [train_conf_image, test_conf_image]
    new_im2 = appendHorizontal(images2, 400)

    final_im = appendVertical([new_im, new_im2], 0)

    bio = BytesIO()
    final_im.save(bio, format='png')

    images[0].close()
    images[1].close()
    new_im.close()
    new_im2.close()
    final_im.close()
    return bio


def showSaveImage(image1, image2, directory, fold, saveFile, average):
    if average:
        image2.seek(0)
        results_image = Image.open(image2)
        results_image.save(directory + (fold + ".png" if isinstance(fold, str)
                                        else ("fold " + str(fold) + ".png")), format='png')
        bio = BytesIO()
        results_image.save(bio, format='png')
        image2.close()
        bio.close()
        return

    image1.seek(0)
    plot_image = Image.open(image1)

    image2.seek(0)
    results_image = Image.open(image2)
    images = [plot_image, results_image]

    big_im = appendVertical(images, 0)

    if saveFile:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

        big_im.save(directory + (fold + ".png" if isinstance(fold, str)
                                 else ("fold " + str(fold) + ".png")), format='png')

    bio = BytesIO()
    big_im.save(bio, format='png')

    image1.close()
    image2.close()
    big_im.close()
    bio.close()

# @title Data Preparation Function Definitions


def ready_data(outputs, filelines):
    N = len(outputs)

    sent_lengths = [len(sentence) for sentence in filelines[0:N]]
    longest_sent = max(sent_lengths)
    sent_lengths = torch.from_numpy(
        np.array(sent_lengths)).to(device, dtype=torch.long)

    padded_vectors = np.zeros((longest_sent, N, 300))
    i = 0
    for line in filelines[0:N]:
        j = 0
        for word in line:
            if word in gloveModel:
                padded_vectors[j, i] = gloveModel[word]
            elif word.lower() in gloveModel:
                padded_vectors[j, i] = gloveModel[word.lower()]
            else:
                padded_vectors[j, i] = np.zeros(300)
            j += 1
        i += 1

    padded_vectors = torch.from_numpy(padded_vectors).to(device, dtype=dtype)

    targets = outputs[0:N]

    return padded_vectors, targets, sent_lengths


def get_batch(batch, batch_size, x, y, sent_len):
    N = len(y)
    beginning = batch * batch_size

    ending = ((batch + 1) * batch_size) if ((batch + 1)
                                            * batch_size) < N else N

    x = x[:, beginning:ending, :]
    y = y[beginning:ending]
    sent_len = sent_len[beginning:ending]

    sent_len, indices = torch.sort(sent_len, descending=True)
    x = x[:, indices, :]
    y = y[indices]

    return x, y, sent_len


def ask_one(sentence, glove_used):
    sentence_length = len(sentence)
    padded_vector = np.zeros((sentence_length, 1, 300))
    sentence_length = torch.tensor(
        [sentence_length]).to(device, dtype=torch.long)
    j = 0
    for word in sentence:
        if word in gloveModel:
            padded_vector[j, 0] = gloveModel[word]
        else:
            padded_vector[j, 0] = np.zeros(300)
        j += 1
    padded_vector = torch.from_numpy(padded_vector).to(device, dtype=dtype)
    return padded_vector, sentence_length


# @title Early Stopping Definition

# https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

# @title Train Function Definition


def train():
    number_of_epochs = 1000
    n_splits = 5
    early_stopping_patience = 15

    outputs, filelines = zip(*filtereddata)
    outputs = torch.LongTensor(outputs).to(device)

    padded_vectors, targets, sent_lengths = ready_data(outputs, filelines)
    targets_on_cpu = targets.cpu()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    i = 0
    train_classification_reports, test_classification_reports = [], []
    all_train_conf_matrices, all_test_conf_matrices = [], []
    for train_index, test_index in skf.split(np.zeros(len(targets)), targets_on_cpu):
        i += 1
        x_train, x_test = padded_vectors[:, train_index,
                                         :], padded_vectors[:, test_index, :]
        y_train, y_test = targets[train_index], targets[test_index]
        sent_len_train, sent_len_test = sent_lengths[train_index], sent_lengths[test_index]
        x_test, y_test, sent_len_test = get_batch(
            0, len(y_test), x_test, y_test, sent_len_test)

        model = LSTM(300, hidden_size, num_layers, bidirectional,
                     batchnorm, dropout_hidden, dropout_output).to(device)
        # https://discuss.pytorch.org/t/vgg-output-layer-no-softmax/9273/5
        loss_function = nn.CrossEntropyLoss().to(device)
        #optimizer = optim.Adam(model.parameters(), lr=0.01)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(
            0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)

        early_stopping = EarlyStopping(
            patience=early_stopping_patience, verbose=True)

        N = len(y_train)

        trainlosses, testlosses = [None], [None]
        fold_train_classification_reports, fold_test_classification_reports = [], []
        fold_train_conf_matrices, fold_test_conf_matrices = [], []
        for epoch in range(1, number_of_epochs + 1):
            ###################
            # train the model #
            ###################
            model.train()  # prep model for training

            shuffleindices = torch.randperm(len(y_train))
            x_train.copy_(x_train[:, shuffleindices, :])
            y_train.copy_(y_train[shuffleindices])
            sent_len_train.copy_(sent_len_train[shuffleindices])
            for batch in range(math.ceil(N / batch_size)):
                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                # get_data gets the data from the dataset (sequence batch, size batch_size)
                x_batch, targets_batch, sent_lengths_batch = get_batch(
                    batch, batch_size, x_train, y_train, sent_len_train)

                # forward pass: compute predicted outputs by passing inputs to the model
                class_pred = model(x_batch, sent_lengths_batch)
                # calculate the loss
                loss = loss_function(class_pred, targets_batch)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()

            """===================================================================================================================="""
            directory = dirname + "results/es_n" + str(n_splits) + "+b" + str(batch_size) + "+e" + str(number_of_epochs) + "+lr" + str(learning_rate) + "+hidden" + str(hidden_size) + "+ly" + str(num_layers) \
                + ("+bd" if bidirectional else "") + ("+bn" if batchnorm else "") + \
                "+dp_h" + str(dropout_hidden) + "+dp_o" + \
                str(dropout_output) + "/"
            fold = i
            saveFile = epoch == number_of_epochs
            """===================================================================================================================="""

            ######################
            # validate the model #
            ######################
            model.eval()  # prep model for evaluation
            """===================================================================================================================="""
            ######################
            #   FIND TEST LOSS   #
            ######################
            # forward pass: compute predicted outputs by passing inputs to the model
            class_pred = model(x_test, sent_len_test)
            # calculate the loss
            loss = loss_function(class_pred, y_test)

            _, pred = class_pred.cpu().detach().max(dim=1)

            testlosses.append(loss.item())  # record validation loss
            minposs = testlosses.index(min(testlosses[1:])) - 1
            y_test2 = y_test.cpu()
            classification_report = metrics.classification_report(
                y_true=y_test2, y_pred=pred, target_names=emotions, output_dict=True, zero_division=0)
            test_conf_matrix = metrics.confusion_matrix(
                y_true=y_test2, y_pred=pred)
            fold_test_classification_reports.append(
                fix_report(classification_report))
            fold_test_conf_matrices.append(test_conf_matrix)

            del classification_report
            del test_conf_matrix
            del loss
            del y_test2

            tobeprintedtest = ["", "", "", "", "", "", "", "", "", "", "", "                         TEST DATA",
                               report_to_string(fold_test_classification_reports[minposs])]
            tobeprintedtest = '\n'.join(tobeprintedtest)
            """===================================================================================================================="""
            """===================================================================================================================="""
            ######################
            # FIND TRAINING LOSS #
            ######################
            x_train2, y_train2, sent_len_train2 = get_batch(
                0, len(y_train), x_train, y_train, sent_len_train)
            # forward pass: compute predicted outputs by passing inputs to the model
            class_pred = model(x_train2, sent_len_train2)
            # calculate the loss
            loss = loss_function(class_pred, y_train2)

            _, pred = class_pred.cpu().detach().max(dim=1)

            trainlosses.append(loss.item())  # record training loss
            y_train2 = y_train2.cpu()
            classification_report = metrics.classification_report(
                y_true=y_train2, y_pred=pred, target_names=emotions, output_dict=True, zero_division=0)
            train_conf_matrix = metrics.confusion_matrix(
                y_true=y_train2, y_pred=pred)
            fold_train_classification_reports.append(
                fix_report(classification_report))
            fold_train_conf_matrices.append(train_conf_matrix)

            del classification_report
            del train_conf_matrix
            del loss
            del x_train2
            del y_train2
            del sent_len_train2

            tobeprintedtrain = ["LEARNING RATE = " + str(learning_rate), "BATCH SIZE = " + str(batch_size), "HIDDEN_SIZE = " + str(hidden_size),
                                str(num_layers) + " LAYERS", "BIDIRECTIONAL " + ("YES" if bidirectional else "NO"), "BATCHNORM " + (
                                    "YES" if batchnorm else "NO"), "DROPOUT HIDDEN = " + str(dropout_hidden),
                                "DROPOUT OUTPUT = " + str(dropout_output),
                                "", "FOLD " + str(fold) + "/" + str(n_splits), "EPOCH " + str(
                                    epoch) + "/" + str(number_of_epochs), "                        TRAIN DATA",
                                report_to_string(fold_train_classification_reports[minposs])]
            tobeprintedtrain = '\n'.join(tobeprintedtrain)
            """===================================================================================================================="""
            """===================================================================================================================="""
            """===================================================================================================================="""
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(testlosses[-1], model)
            """===================================================================================================================="""
            """===================================================================================================================="""
            if saveFile or early_stopping.early_stop:
                image1 = plottraintest(trainlosses, testlosses)
                image2 = showResults(tobeprintedtrain, tobeprintedtest,
                                     fold_train_conf_matrices[minposs], fold_test_conf_matrices[minposs], emotions)
                showSaveImage(image1, image2, directory, fold,
                              saveFile or early_stopping.early_stop, False)

                train_classification_reports.append(
                    fold_train_classification_reports[minposs])
                all_train_conf_matrices.append(
                    fold_train_conf_matrices[minposs])
                fold_train_conf_matrices = []
                test_classification_reports.append(
                    fold_test_classification_reports[minposs])
                fold_test_classification_reports = []
                all_test_conf_matrices.append(fold_test_conf_matrices[minposs])
            """===================================================================================================================="""
            if n_splits == i and (epoch == number_of_epochs or early_stopping.early_stop):
                train_conf_matrices_average = np.round(
                    np.mean(all_train_conf_matrices, axis=0), 1)
                test_conf_matrices_average = np.round(
                    np.mean(all_test_conf_matrices, axis=0), 1)

                train_classification_average = report_average(
                    train_classification_reports)
                test_classification_average = report_average(
                    test_classification_reports)

                tobeprintedtrain = ["LEARNING RATE = " + str(learning_rate), "BATCH SIZE = " + str(batch_size), "HIDDEN_SIZE = " + str(hidden_size),
                                    str(num_layers) + " LAYERS", "BIDIRECTIONAL " + ("YES" if bidirectional else "NO"), "BATCHNORM " + (
                                        "YES" if batchnorm else "NO"), "DROPOUT HIDDEN = " + str(dropout_hidden),
                                    "DROPOUT OUTPUT = " + str(dropout_output),
                                    "", str(fold) + " FOLD AVERAGE", "EPOCH " +
                                    str(epoch), "                        TRAIN DATA",
                                    train_classification_average]
                tobeprintedtrain = '\n'.join(tobeprintedtrain)

                tobeprintedtest = ["", "", "", "", "", "", "", "", "", "", "", "                         TEST DATA",
                                   test_classification_average]
                tobeprintedtest = '\n'.join(tobeprintedtest)

                averageImage2 = showResults(
                    tobeprintedtrain, tobeprintedtest, train_conf_matrices_average, test_conf_matrices_average, emotions)
                showSaveImage(None, averageImage2, directory, str(fold) + " fold average",
                              n_splits == i and (epoch == number_of_epochs or early_stopping.early_stop), True)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            """===================================================================================================================="""
            print('epoch: {} memory use: {}MB'.format(
                epoch, torch.cuda.memory_allocated()/2.**20))
        torch.cuda.empty_cache()
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))
        torch.save(model.state_dict(), dirname + "models/es_n" + str(i) + "+b" + str(batch_size) + "+e" + str(number_of_epochs) + "+lr" + str(learning_rate) + "+hidden" + str(hidden_size) + "+ly" + str(num_layers)
                   + ("+bd" if bidirectional else "") + ("+bn" if batchnorm else "") + "+dp_h" + str(dropout_hidden) + "+dp_o" + str(dropout_output) + ".pth")

        del x_train
        del x_test
        del y_train
        del y_test
        del pred
        del sent_len_train
        del sent_len_test
        del trainlosses
        del testlosses
        del loss_function
        del optimizer
        del early_stopping
        del model
        gc.collect()

        """
        model.eval()
        sentence = list(parser.tokenize("I was going home when I saw him"))
        input1, sent_length = ask_one(sentence, glove_used)
        class_pred = model(input1, sent_length)
        print(class_pred)
        _, prediction = class_pred.max(dim=1)
        print(emotions[prediction[0]])
        """


lrlist = [0.003]
batchsizelist = [16]
hdlist = [256]
lylist = [2]
bdlist = [False]
bnlist = [False]
dphlist = [0.3]
dpolist = [0.9]


# @title Training
# torch.set_printoptions(precision=25)
iterlist = itertools.product(
    lrlist, batchsizelist, hdlist, lylist, bdlist, bnlist, dphlist, dpolist)


for learning_rate, batch_size, hidden_size, num_layers, bidirectional, batchnorm, dropout_hidden, dropout_output in iterlist:
    if num_layers == 1 and dropout_hidden != 0.0:
        continue
    """%whos
    print("==============================")"""
    train()
    """%whos
    print("==============================")"""
# torch.set_printoptions(precision=4)
