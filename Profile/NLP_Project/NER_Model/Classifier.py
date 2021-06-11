import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_dim, n_class):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.n_class = n_class

        if self.n_class == 2:
            # Binary Classification
            self.mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=input_dim // 2),
                nn.ReLU(),
                nn.Linear(in_features=input_dim // 2, out_features=1)
            )  # .to(device)

            self.loss_fn = nn.BCEWithLogitsLoss()  # .to(device)
            self.last = F.sigmoid
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=input_dim // 2),
                nn.ReLU(),
                nn.Linear(in_features=input_dim // 2, out_features=self.n_class)
            )  # .to(device)
            self.loss_fn = nn.CrossEntropyLoss()  # .to(device)
            self.last = F.softmax

    def forward(self, x):
        x = self.mlp(x)
        return x

    def loss(self, x, label):
        """
        Args:
          1. x (batch_size,seq_len,input_dim)
          2. label(batch_size,seq_len,1)
        """
        x = self.mlp(x)
        # print(pred)
        loss = self.loss_fn(x.squeeze(), label)

        return loss