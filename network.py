import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size,
                 encoder_hidden_size, decoder_hidden_size,
                 encoder_layers=1, decoder_layers=1,
                 dropout=0.1, embedding_weights=None, device=None):
        super(Model, self).__init__()

        self.unk_id = 0
        self.pad_id = 1
        self.go_id = 2
        self.eos_id = 3

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.dropout = dropout
        self.embedding_weights = embedding_weights
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = Embedding(vocab_size=self.vocab_size,
                                   embedding_dim=self.embedding_dim,
                                   weights=self.embedding_weights,
                                   device=self.device)

        self.encoder = Encoder(hidden_size=self.encoder_hidden_size,
                               n_layers=self.encoder_layers,
                               dropout=self.dropout,
                               device=self.device)

        self.decoder = Decoder(hidden_size=self.decoder_hidden_size,
                               output_size=self.output_size,
                               n_layers=self.decoder_layers,
                               dropout=self.dropout,
                               device=self.device)
        self.attn = self.decoder.attn

    def forward(self, encoder_input, encoder_length, decoder_input, decoder_length):
        encoder_input = self.embedding(encoder_input)
        decoder_input = self.embedding(decoder_input)
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_length)
        decoder_outputs, decoder_hidden = self.decoder(decoder_input,
                                                       decoder_length,
                                                       encoder_outputs,
                                                       encoder_length,
                                                       encoder_hidden)
        return decoder_outputs

    def predict(self, encoder_input, encoder_length=None):
        if type(encoder_input) in (list, np.ndarray):
            encoder_input = torch.LongTensor(encoder_input)
        if type(encoder_length) == int:
            self.encoder_length = torch.LongTensor([encoder_length])

        if len(encoder_input.shape) == 1:
            encoder_input = encoder_input.view(1, -1)

        self.encoder_input = encoder_input
        if encoder_length is None:
            length = len(encoder_input[encoder_input > 3])
            self.encoder_length = torch.LongTensor([length])
        else:
            self.encoder_length = encoder_length
        decoder_input = torch.LongTensor([[self.go_id]]).to(self.device)
        self.decoder_length = torch.LongTensor([1]).to(self.device)
        self.encoder_input = self.embedding(encoder_input.to(self.device))
        decoder_input = self.embedding(decoder_input.to(self.device))
        self.encoder_outputs, self.encoder_hidden = self.encoder(self.encoder_input.to(self.device),
                                                                 self.encoder_length.to(self.device))
        self.decoder_outputs, self.decoder_hidden = self.decoder(decoder_input,
                                                                 self.decoder_length,
                                                                 self.encoder_outputs,
                                                                 self.encoder_length,
                                                                 self.encoder_hidden)
        return self.decoder_outputs

    def next(self, next_input):
        if type(next_input) in (list, np.ndarray):
            next_input = torch.LongTensor(next_input)
        if type(next_input) == int:
            next_input = torch.LongTensor([[next_input]])

        decoder_input = self.embedding(next_input.to(self.device))
        self.decoder_outputs, self.decoder_hidden = self.decoder(decoder_input,
                                                                 self.decoder_length,
                                                                 self.encoder_outputs,
                                                                 self.encoder_length,
                                                                 self.decoder_hidden)
        return self.decoder_outputs


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, weights=None, device=None):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if weights is None:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.device)
        else:
            self.embedding = nn.Embedding(self.vocab_size,
                                          self.embedding_dim).from_pretrained(weights).to(self.device)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, X):
        y = self.embedding(X)
        return y


class Encoder(nn.Module):
    def __init__(self, hidden_size, n_layers=1, dropout=0.1, device=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gru = nn.GRU(self.hidden_size,
                          self.hidden_size,
                          self.n_layers,
                          dropout=self.dropout,
                          batch_first=True).to(self.device)

    def forward(self, input_seqs, input_lengths, hidden=None):
        packed = nn.utils.rnn.pack_padded_sequence(input_seqs, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, device=None):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.W = nn.Linear(self.hidden_size * 2, self.hidden_size).to(self.device)
        self.v = nn.Linear(self.hidden_size, 1).to(self.device)
        self.dropout = nn.Dropout(p=dropout).to(self.device)

    def forward(self, encoder_outputs, decoder_outputs, encoder_length, decoder_length):
        batch_size = encoder_outputs.size(0)
        encoder_max_len = encoder_outputs.size(1)
        decoder_max_len = decoder_outputs.size(1)

        # self.attention = torch.zeros((batch_size, decoder_length.max(), encoder_length.max())).to(self.device)
        # for i in range(batch_size):
        #     for j in range(decoder_length[i]):
        #         score_j = torch.zeros(encoder_length[i]).to("cuda")
        #         for k in range(encoder_length[i]):
        #             a = self.v(self.W(torch.cat((encoder_outputs[i, k], decoder_outputs[i, j])))).view(-1)
        #             score_j[k] = a
        #         # print(score_j)
        #         # print(F.softmax(score_j))
        #         # print()
        #         self.attention[i, j, :k + 1] = F.softmax(score_j)
        # encoder_outputs = encoder_outputs.view((batch_size, 1, encoder_max_len, -1))
        # self.encoder_outputs = encoder_outputs.repeat(1, decoder_max_len, 1, 1)
        # context = (self.attention.view(batch_size, decoder_length.max(), encoder_length.max(),
        #                                1) * self.encoder_outputs).sum(dim=2)

        mask = torch.ones((batch_size, decoder_max_len, encoder_max_len, 1)).to(self.device)
        for i, j in enumerate(encoder_length):
            mask[i, :, j:, :] = 0
        for i, j in enumerate(decoder_length):
            mask[i, j:, :, :] = 0

        encoder_outputs = encoder_outputs.view((batch_size, 1, encoder_max_len, -1))
        encoder_outputs = encoder_outputs.repeat(1, decoder_max_len, 1, 1)
        encoder_outputs = encoder_outputs * mask
        decoder_outputs = decoder_outputs.view((batch_size, decoder_max_len, 1, -1))
        decoder_outputs = decoder_outputs.repeat(1, 1, encoder_max_len, 1)
        decoder_outputs = decoder_outputs * mask

        score = F.tanh(self.W(torch.cat((encoder_outputs, decoder_outputs), dim=3)))
        energy = self.v(score)
        a = F.softmax(energy, dim=2)

        a_with_mask = a * mask
        a_with_mask_sum = a_with_mask.sum(dim=2).view(batch_size, decoder_max_len)
        a_with_mask_sum[a_with_mask_sum == 0] = 1
        self.attention = a_with_mask / a_with_mask_sum.view(batch_size, decoder_max_len, 1, 1)
        context = (self.attention * encoder_outputs).sum(dim=2)
        context = self.dropout(context)

        return context


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1, device=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.attn = Attn(self.hidden_size,
                         self.dropout,
                         self.device)

        self.gru = nn.GRU(self.hidden_size,
                          self.hidden_size,
                          self.n_layers,
                          dropout=self.dropout,
                          batch_first=True).to(self.device)

        self.dense = nn.Linear(hidden_size * 2, hidden_size).to(self.device)
        self.output = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, input_seqs, input_lengths, encoder_outputs, encoder_length, hidden=None):
        packed = nn.utils.rnn.pack_padded_sequence(input_seqs, input_lengths, batch_first=True, enforce_sorted=False)
        decoder_outputs, hidden = self.gru(packed, hidden)
        decoder_outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(decoder_outputs, batch_first=True)
        context = self.attn(encoder_outputs, decoder_outputs, encoder_length, input_lengths)
        outputs = torch.cat((decoder_outputs, context), dim=2)
        dense = F.tanh(self.dense(outputs))
        logists = self.output(dense)

        return logists, hidden


