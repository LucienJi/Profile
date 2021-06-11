
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class Highway(torch.nn.Module):
    """
    Function:
      1. Output = g * X + (1-g)*f(X)
    # Parameters
    input_dim : `int`, required
        The dimensionality of :math:`x`.  We assume the input has shape `(batch_size, ...,
        input_dim)`.
    num_layers : `int`, optional (default=`1`)
        The number of highway layers to apply to the input.
    """

    def __init__(
            self,
            input_dim,
            num_layers,
            activation=torch.nn.functional.relu,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class CharEmb(nn.Module):
    def __init__(self,
                 char_size,
                 max_word_size,
                 emb_size,
                 hidden_size,
                 device,
                 use_lstm):
        super(CharEmb, self).__init__()
        self.char_size = char_size
        self.max_word_size = max_word_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.device = device
        self.use_lstm = use_lstm
        # Char Emb
        self.char_emb = nn.Embedding(char_size, emb_size)
        self.filter = [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 1024]]
        self._init_emb()
        self._init_conv()
        self._init_highway()
        self._init_projection()

        if self.use_lstm:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size // 2, num_layers=2, bidirectional=True)

    def init_hidden(self, batch_size, device):
        return (torch.randn(4, batch_size, self.hidden_size // 2).to(device),
                torch.randn(4, batch_size, self.hidden_size // 2).to(device))

    def _init_emb(self):
        r = 0.5 / self.char_size
        self.char_emb.weight.data.uniform_(-r, r).to(self.device)

    def _init_conv(self):
        self._convolutions = nn.ModuleList()
        for i, (width, num) in enumerate(self.filter):
            conv = torch.nn.Conv1d(
                in_channels=self.emb_size, out_channels=num,
                kernel_size=width, bias=True, ).to(self.device)
            self._convolutions.append(conv)

    def _init_highway(self):
        # the highway layers have same dimensionality as the number of cnn filters
        n_filters = sum(f[1] for f in self.filter)
        n_highway = 2
        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)

    def _init_projection(self):
        n_filters = sum(f[1] for f in self.filter)
        self._projection = torch.nn.Linear(n_filters, self.hidden_size, bias=True)

    def forward(self, sentence):
        # sentence : (batch_size,seq_leq,word_length)
        bz = sentence.size(0)
        seq_len = sentence.size(1)
        sentence = sentence.view(bz * seq_len, self.max_word_size)
        emb = self.char_emb(sentence).transpose(1, 2)  # (batch_size*seq_len,emb_size,word_length)
        convs = []
        for i in range(len(self._convolutions)):
            conv = self._convolutions[i]
            conved = conv(emb)  # (bathc_size,seq_len,Channel_out,filtered size)
            conved, _ = torch.max(conved, dim=-1)  # (bathc_size,seq_len,Channel_out)
            conved = F.relu(conved)
            convs.append(conved)

        final_emb = torch.cat(convs, dim=-1)  # (bathc_size,seq_len,total_Channel_out)
        final_emb = self._highways(final_emb)  # (bathc_size,seq_len,total_Channel_out)
        final_emb = self._projection(final_emb)  # (bathc_size,seq_len,hidden_size)
        final_emb = final_emb.view(bz, seq_len, -1)
        if self.use_lstm:
            self.hiddens = self.init_hidden(bz, sentence.device)
            lstm_out, self.hiddens = self.lstm(final_emb.transpose(0, 1), self.hiddens)
            return lstm_out.transpose(0, 1)
        else:
            return final_emb