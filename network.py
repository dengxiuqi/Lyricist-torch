import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size, encoder_hidden_size, decoder_hidden_size,
                 encoder_layers=1, decoder_layers=1, dropout=0.1, embedding_weights=None, device=None):
        """
        Encoder-Decoder-Attention网络
        :param vocab_size: Embedding层输入词典的大小, Embedding层的第一维大小
        :param embedding_dim: Embedding层输出维度, Embedding层的第二维大小
        :param output_size: 网络输出维度
        :param encoder_hidden_size: 编码器RNN的隐藏层大小
        :param decoder_hidden_size: 加密器RNN的隐藏层大小
        :param encoder_layers: 编码器RNN的层数
        :param decoder_layers: 解码器RNN的层数
        :param embedding_weights: 预训练的词向量权重
        :param device: "cpu"/"cuda"
        """
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
        # Embedding层
        self.embedding = Embedding(vocab_size=self.vocab_size,
                                   embedding_dim=self.embedding_dim,
                                   weights=self.embedding_weights,
                                   device=self.device)
        # Encoder
        self.encoder = Encoder(hidden_size=self.encoder_hidden_size,
                               n_layers=self.encoder_layers,
                               dropout=self.dropout,
                               device=self.device)
        # Decoder
        self.decoder = Decoder(hidden_size=self.decoder_hidden_size,
                               output_size=self.output_size,
                               n_layers=self.decoder_layers,
                               dropout=self.dropout,
                               device=self.device)
        # Attention(Decoder内部自带Attention层)
        self.attn = self.decoder.attn

    def forward(self, encoder_input, encoder_length, decoder_input, decoder_length):
        """
        :param encoder_input:  编码器输入, shape: (1, time_step, word_id)
        :param encoder_length: 编码器输入文本有效长度, shape: (1, )
        :param decoder_input:  解码器输入, shape: (1, time_step, word_id)
        :param decoder_length: 解码器输入文本有效长度, shape: (1, )
        """
        encoder_input = self.embedding(encoder_input)
        decoder_input = self.embedding(decoder_input)
        encoder_outputs, encoder_hidden = self.encoder(encoder_input,
                                                       encoder_length)
        decoder_outputs, decoder_hidden = self.decoder(decoder_input,
                                                       decoder_length,
                                                       encoder_outputs,
                                                       encoder_length,
                                                       encoder_hidden)
        return decoder_outputs

    def predict(self, encoder_input, encoder_length=None):
        """
        用于test阶段, 根据encoder_input, 通过解码器生成一个词
        :param encoder_input:  编码器输入, shape: (batch_size, time_step, word_id)
        :param encoder_length: 编码器输入文本有效长度, shape: (batch_size, )
        """
        # 自动计算输入编码器文本有效长度
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
        # 自动生成解码器的输入数据: ["<go>"]
        decoder_input = torch.LongTensor([[self.go_id]]).to(self.device)
        self.decoder_length = torch.LongTensor([1]).to(self.device)
        self.encoder_input = self.embedding(encoder_input.to(self.device))
        decoder_input = self.embedding(decoder_input.to(self.device))
        # 生成一个词
        self.encoder_outputs, self.encoder_hidden = self.encoder(self.encoder_input.to(self.device),
                                                                 self.encoder_length.to(self.device))
        self.decoder_outputs, self.decoder_hidden = self.decoder(decoder_input,
                                                                 self.decoder_length,
                                                                 self.encoder_outputs,
                                                                 self.encoder_length,
                                                                 self.encoder_hidden)
        return self.decoder_outputs

    def next(self, next_input):
        """
        用于test阶段, 在已用predict预测生成过一个词后, 输入该词, 继续预测下一个词
        :param next_input: 解码器在当前时刻的输入, shape: (1, 1, word_id)
        """
        # 自动计算输入解码器文本有效长度
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
            p.requires_grad = False     # Embedding用的是预训练的Word2Vec, 不参与网络训练

    def forward(self, X):
        y = self.embedding(X)
        return y


class Encoder(nn.Module):
    def __init__(self, hidden_size, n_layers=1, dropout=0.1, device=None):
        """
        编码器
        :param hidden_size: RNN的隐藏层大小
        :param n_layers: RNN的层数
        :param device: "cpu"/"cuda"
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 用于编码的GRU, 也可以使用普通RNN和LSTM
        self.gru = nn.GRU(self.hidden_size,
                          self.hidden_size,
                          self.n_layers,
                          dropout=self.dropout,
                          batch_first=True).to(self.device)

    def forward(self, input_seqs, input_lengths, hidden=None):
        """
        :param input_seqs: 编码器输入, shape: (batch_size, time_step, word_id)
        :param input_lengths: 编码器输入文本有效长度, shape: (batch_size, )
        :param hidden: Bool类型, 是否返回RNN的hidden state
        """
        packed = nn.utils.rnn.pack_padded_sequence(input_seqs, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, device=None):
        """
        注意力
        :param hidden_size: 编码器和解码器的隐藏层大小
        :param device: "cpu"/"cuda"
        """
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 用于计算注意力得分的两个参数W和v
        self.W = nn.Linear(self.hidden_size * 2, self.hidden_size).to(self.device)
        self.v = nn.Linear(self.hidden_size, 1).to(self.device)
        self.dropout = nn.Dropout(p=dropout).to(self.device)

    def forward(self, encoder_outputs, decoder_outputs, encoder_length, decoder_length):
        """
        :param encoder_outputs: 编码器的输出, shape: (batch, time_step, hidden_size)
        :param decoder_outputs: 解码器的输出, shape: (batch, time_step, hidden_size)
        :param encoder_length: 编码器输入文本有效长度, shape: (batch_size, )
        :param decoder_length: 解码器输入文本有效长度, shape: (batch_size, )
        """
        batch_size = encoder_outputs.size(0)
        encoder_max_len = encoder_outputs.size(1)
        decoder_max_len = decoder_outputs.size(1)

        # 直观的注意力权重计算
        # self.attention = torch.zeros((batch_size, decoder_length.max(), encoder_length.max())).to(self.device)
        # for i in range(batch_size):
        #     for j in range(decoder_length[i]):
        #         score_j = torch.zeros(encoder_length[i]).to("cuda")
        #         for k in range(encoder_length[i]):
        #             score = self.W(torch.cat((encoder_outputs[i, k], decoder_outputs[i, j])))
        #             score = F.tanh(score)
        #             a = self.v(score).view(-1)
        #             score_j[k] = a
        #         self.attention[i, j, :k + 1] = F.softmax(score_j)
        # encoder_outputs = encoder_outputs.view((batch_size, 1, encoder_max_len, -1))
        # self.encoder_outputs = encoder_outputs.repeat(1, decoder_max_len, 1, 1)
        # context = (self.attention.view(batch_size, decoder_length.max(), encoder_length.max(),
        #                                1) * self.encoder_outputs).sum(dim=2)

        # 并行计算所有注意力, 提高网络效率
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
        # 计算注意力得分
        score = F.tanh(self.W(torch.cat((encoder_outputs, decoder_outputs), dim=3)))
        energy = self.v(score)
        a = F.softmax(energy, dim=2)
        # Encoder中超过有效长度的部分不参与Attention计算
        a_with_mask = a * mask
        a_with_mask_sum = a_with_mask.sum(dim=2).view(batch_size, decoder_max_len)
        a_with_mask_sum[a_with_mask_sum == 0] = 1
        self.attention = a_with_mask / a_with_mask_sum.view(batch_size, decoder_max_len, 1, 1)
        # 加权求和
        context = (self.attention * encoder_outputs).sum(dim=2)
        context = self.dropout(context)
        return context


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=0.1, device=None):
        """
        解码器
        :param hidden_size: RNN的隐藏层大小
        :param n_layers: RNN的层数
        :param device: "cpu"/"cuda"
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Attention
        self.attn = Attn(self.hidden_size,
                         self.dropout,
                         self.device)
        # 用于解码的GRU
        self.gru = nn.GRU(self.hidden_size,
                          self.hidden_size,
                          self.n_layers,
                          dropout=self.dropout,
                          batch_first=True).to(self.device)
        # 两个全连接层
        self.dense = nn.Linear(hidden_size * 2, hidden_size).to(self.device)
        self.output = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, input_seqs, input_lengths, encoder_outputs, encoder_length, hidden=None):
        """
        :param input_seqs: 编码器输入, shape: (batch_size, time_step, word_id)
        :param input_lengths: 编码器输入文本有效长度, shape: (batch_size, )
        :param encoder_outputs: 编码器的输出, shape: (batch, time_step, hidden_size)
        :param encoder_length: 编码器输入文本有效长度, shape: (batch_size, )
        :param hidden: Bool类型, 是否返回RNN的hidden state
        """
        # 解码
        packed = nn.utils.rnn.pack_padded_sequence(input_seqs, input_lengths, batch_first=True, enforce_sorted=False)
        decoder_outputs, hidden = self.gru(packed, hidden)
        decoder_outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(decoder_outputs, batch_first=True)
        # 注意力
        context = self.attn(encoder_outputs, decoder_outputs, encoder_length, input_lengths)
        # 拼接
        outputs = torch.cat((decoder_outputs, context), dim=2)
        # 两个全连接
        dense = F.tanh(self.dense(outputs))
        logists = self.output(dense)

        return logists, hidden


