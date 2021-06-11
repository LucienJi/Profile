import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class WordEmb(nn.Module):
    def __init__(self,
                 voc_size,
                 emb_size,
                 hidden_size,
                 device,
                 use_lstm):
        super(WordEmb, self).__init__()
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.hidden_dim = hidden_size
        self.device = device
        self.use_lstm = use_lstm

        ## Words_emb
        self.words_emb = nn.Embedding(self.voc_size, self.emb_size)
        self._init_emb()
        ## LSTM
        if self.use_lstm:
            self.lstm = nn.LSTM(self.emb_size, self.hidden_dim // 2, num_layers=2, bidirectional=True)
        else:
            self._projection = nn.Linear(emb_size, hidden_size)

    def _init_emb(self):
        r = 0.5 / self.voc_size
        self.words_emb.weight.data.uniform_(-r, r).to(self.device)

    def init_hidden(self, batch_size, device):
        return (torch.randn(4, batch_size, self.hidden_dim // 2).to(device),
                torch.randn(4, batch_size, self.hidden_dim // 2).to(device))

    def forward(self, sentences: torch.Tensor):
        """
        Arg:
          sentence: (batch_size,seq_len,words_idx)

        Output:

          lstm_out: (batch_size,seq_len,hidden_size)
        """
        embs = self.words_emb(sentences)

        embs = embs.transpose(0, 1)
        if self.use_lstm:
            batch_size = sentences.size(0)
            self.hiddens = self.init_hidden(batch_size, sentences.device)
            # print(embs.shape,self.hiddens)
            lstm_out, self.hiddens = self.lstm(embs, self.hiddens)
            return lstm_out.transpose(0, 1)
        else:
            return self._projection(embs)

    def eval(self, sentences: torch.Tensor):
        """
        Arg:
          sentence: (batch_size,seq_len,words_idx)

        Output:

          lstm_out: (batch_size,seq_len,hidden_size)
        """
        embs = self.words_emb(sentences).view(sentences.size(1), sentences.size(0), self.emb_size)
        self.hiddens = self.init_hidden(sentences.size(0))
        lstm_out, self.hiddens = self.lstm(embs, self.hiddens)
        return lstm_out.transpose(0, 1)
