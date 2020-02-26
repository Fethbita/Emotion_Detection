# @title Model Definition
import torch
import torch.nn as nn
device = torch.device("cuda:0")


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, batchnormactive, dropout_hidden, dropout_output):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batchnormactive = batchnormactive

        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=num_layers,
                            bidirectional=self.bidirectional, dropout=dropout_hidden)
        self.lstm.to(device)

        self.batchnormlayer = None
        if self.batchnormactive:
            self.batchnormlayer = nn.BatchNorm1d(
                self.hidden_size * 2 if self.bidirectional else self.hidden_size)
            self.batchnormlayer.to(device)

        self.dropoutlayer = nn.Dropout(dropout_output)
        self.dropoutlayer.to(device)

        self.fc = nn.Linear(self.hidden_size *
                            2 if self.bidirectional else self.hidden_size, 6)
        self.fc.to(device)

    def forward(self, glove_vec, sent_lengths):
        # glove_vec.shape = (sentence_len, batch_size, 300)
        output = torch.nn.utils.rnn.pack_padded_sequence(
            glove_vec, sent_lengths)
        # packed sequence
        output, _ = self.lstm(output)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        # padded sequence
        # output.shape = (sentence_len, batch_size, hidden_size * 2 if bidirectional else hidden_size)

        # https://blog.nelsonliu.me/2018/01/24/extracting-last-timestep-outputs-from-pytorch-rnns/
        idx = (sent_lengths - 1).view(-1, 1).expand(-1, self.hidden_size *
                                                    2 if self.bidirectional else self.hidden_size).unsqueeze(0)
        output = output.gather(0, idx).squeeze(0)
        # output.shape = (batch_size, hidden_size * 2 if bidirectional else hidden_size)

        if self.batchnormactive:
            output = self.batchnormlayer(output)

        output = self.dropoutlayer(output)

        output = self.fc(output)
        # output.shape = (batch_size, 8)
        return output
