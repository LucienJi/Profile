import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class PositionEncoding(nn.Module):
    def __init__(self, input_dim, device, dropout=0.1, max_len=100):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, input_dim).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_dim, 2).float() * (-math.log(10000.0) / input_dim))
        position = position.to(device)
        div_term = div_term.to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # if x.device is not self.pe.device:
        #  self.pe.to(x.device)
        """
        Arg:
          x = (batch_size,seq_len,input_dim)
        """
        x = x + self.pe[:, x.size(1), :]
        return self.dropout(x)


"""
class LayerNormalization(nn.Module):
  def __init__(self,input_dim,device,eps = 1e-6):
    super(LayerNormalization,self).__init__()
    self.alpha = torch.nn.Parameter(torch.ones(input_dim)).to(device)
    self.beta = torch.nn.Parameter(torch.zeros(input_dim)).to(device)
    self.eps = eps
  def forward(self,x):
    mean = x.mean(-1,keepdim=True)
    std = x.std(-1,unbiased = False,keepdim=True)
    return self.alpha * (x - mean)/(std + self.eps) + self.beta
"""


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.f2(self.dropout(F.relu(self.f1(x))))


class Attention(nn.Module):
    def __init__(self, input_dim, num_head=1, dropout=0.1):
        super(Attention, self).__init__()
        assert input_dim % num_head == 0
        self.d_k = input_dim // num_head
        self.num_head = num_head
        self.dropout = dropout
        self.FC = nn.Linear(input_dim, input_dim)

    def attention(self, query, key, value, dropout=None):
        dim = query.size(-1)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim)
        coef = F.softmax(score, dim=-1)
        z = torch.matmul(coef, value)
        return z, coef

    def forward(self, query, key, value):
        bz = query.size(0)
        query = query.view(bz, -1, self.num_head, self.d_k).transpose(1, 2)
        key = key.view(bz, -1, self.num_head, self.d_k).transpose(1, 2)
        value = value.view(bz, -1, self.num_head, self.d_k).transpose(1, 2)
        z, coef = self.attention(query, key, value, self.dropout)

        z = z.transpose(1, 2).contiguous().view(bz, -1, self.num_head * self.d_k)
        return self.FC(z), coef


class Encoder(nn.Module):
    def __init__(self, input_dim, max_len, device, num_head=1):
        super(Encoder, self).__init__()
        self.positionencoder = PositionEncoding(input_dim, device, dropout=0.1, max_len=max_len * 2)  # .to(device)
        # self.norm = LayerNormalization(input_dim=input_dim,device = device,eps = 1e-6).to(device)
        self.norm = nn.LayerNorm(normalized_shape=[input_dim])
        self.attention = Attention(input_dim, num_head=num_head, dropout=0.1)  # .to(device)
        self.feedforward = FeedForward(input_dim, hidden_dim=input_dim // 2)  # .to(device)

        self.value_generator = nn.Linear(in_features=input_dim, out_features=input_dim)  # .to(device)
        self.query_generator = nn.Linear(in_features=input_dim, out_features=input_dim)  # .to(device)
        self.key_generator = nn.Linear(in_features=input_dim, out_features=input_dim)  # .to(device)

    def get_attn_coef(self, x):
        x = self.positionencoder(x)

        v = self.value_generator(x)
        q = self.query_generator(x)
        k = self.key_generator(x)
        return self.attention(q, k, v)[1]

    def forward(self, x):
        """
        Args:
          x : (batch_size, seq_len, input_dim)
          return (batch_size,seq_len,input_dim)

        """
        x = self.positionencoder(x)

        v = self.value_generator(x)
        q = self.query_generator(x)
        k = self.key_generator(x)

        x = self.norm(x + self.attention(q, k, v)[0])
        x = self.norm(x + self.feedforward(x))
        return x



