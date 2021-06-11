import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from NER_Model.WordEmbedding import WordEmb
from NER_Model.CharEmbedding import CharEmb
from NER_Model.Classifier import Classifier
from NER_Model.AttentionEncoder import Encoder
from NER_Model.DataSet import Dataset


class Ner_Labeler:
    def __init__(self,
                 model,
                 data):
        self.model = model
        self.data = data

    def test(self, w, c, l, r, max_batch_size=128, device=torch.device("cuda")):
        w, c, l, r = self.data.sample_test(w, c, l, r, max_batch_size)
        confusion_mat = ConfusionMatrix()
        num_test = len(w)
        with torch.no_grad():
            for i in range(num_test):
                words, chars, labels = w[i], c[i], l[i]
                words = torch.LongTensor(words).to(device)
                chars = torch.LongTensor(chars).to(device)
                labels = torch.tensor(labels).to(device)
                predictions = self.model.predict(words, chars)
                error_rate = torch.sum(torch.abs(labels - predictions)) / labels.size(0)
                l_np = labels.squeeze().detach().cpu().numpy()
                p_np = predictions.squeeze().detach().detach().cpu().numpy()
                confusion_mat.addEstimate(l_np.astype(np.int16), p_np.astype(np.int16))
        return confusion_mat

    def train(self, iteration, batch_size, path, augment=True, device=torch.device("cuda"), test_example=None):
        initial_lr = 0.001
        num_sentence = self.data.totalDataLength
        batch_count = iteration * num_sentence / batch_size
        process_bar = tqdm(range(int(batch_count)))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr)
        self.model.to(device)
        self.model.to_device(device)
        loss_list = []
        mat_list = []
        if test_example != None:
            w, c, l, r = self.data.transform(test_example)
        for i in process_bar:
            words, chars, labels = self.data.sample(batch_size, augment)
            optimizer.zero_grad()
            words = torch.LongTensor(words).to(device)
            chars = torch.LongTensor(chars).to(device)
            labels = torch.tensor(labels, dtype=torch.float).to(device)
            loss = self.model.loss(words, chars, labels)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            process_bar.set_description("Loss: %0.8f, lr: %0.6f" % (loss, optimizer.param_groups[0]['lr']))
            # 学习率慢慢下降
            if i * batch_size % 10000 == 0 and i > 0:
                torch.save(self.model.state_dict(), path + "_TEMP")
                lr = initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if (test_example != None) and i % 100 == 0:
                mat_list.append(self.test(w, c, l, r,max_batch_size=32,device=device))
        torch.save(self.model.state_dict(), path + "_FINAL")
        if test_example != None:
            print("### PLOT ###")
            self.plot_loss_confusion(loss_list, mat_list, 10)
        return loss_list, mat_list

    def load(self, path, device=torch.device("cpu")):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)

    def eval(self, sentence: str):
        words = sentence.split(" ")
        n_word = len(words)  # not count start and end
        xWords, xCharacters = self.data.eval([words])
        xWords = torch.tensor(xWords)
        xCharacters = torch.tensor(xCharacters)
        self.model.to(torch.device("cpu"))
        self.model.eval()
        prediction = self.model.predict(xWords, xCharacters).squeeze()
        return prediction[1:1 + n_word]

    def plot_loss_confusion(self, loss_list, mat_list, interval=10):
        fig = plt.figure()
        f_score = []
        recall = []
        for i in range(len(mat_list)):
            f_score.append(mat_list[i].getFscore())
            recall.append(mat_list[i].getRecall())
        print(f_score)
        print(recall)
        axis_x = [(i + 1) * interval for i in range(len(f_score))]
        plt.plot(np.arange(len(loss_list)), loss_list, 'r', label='Train Loss')
        plt.plot(axis_x, f_score, 'b', label='Test F-score')
        plt.plot(axis_x, recall, 'g', label='Test Recall Rate')
        plt.title('Train & Validation')
        plt.ylabel('Y-Axis')
        plt.xlabel('Iteration')
        plt.legend()
        plt.grid()
        plt.show()

    def representation(self, example,device = torch.device("cuda")):
        w, c, l, r = self.data.transform(example)
        w, c, l, r = self.data.sample_test(w, c, l, r)
        n_sample = len(w)
        final_repres = []
        final_words = []
        final_labels = []
        with torch.no_grad():
            for i in range(n_sample):
                words, chars, labels, raws = w[i], c[i], l[i], r[i]
                words = torch.LongTensor(words).to(device)
                chars = torch.LongTensor(chars).to(device)
                repres = self.model.get_representation(words, chars)  # numpy#(batach,max_words_num,repres_dim)
                assert len(raws) == repres.shape[0]
                for j in range(repres.shape[0]):
                    final_words.extend(raws[j][1:-1])
                    final_repres.extend(repres[j][1:1 + len(raws[j][1:-1])])
                    final_labels.extend(labels[j][1:1 + len(raws[j][1:-1])])
        final_words = np.array(final_words)
        final_repres = np.array(final_repres)
        final_labels = np.array(final_labels)
        return final_repres, final_words, final_labels

class NER_Model(nn.Module):
    def __init__(self,
                 voc_size,
                 word_emb_size,
                 char_size,
                 char_emb_size,
                 max_word_len,
                 max_seq_len,
                 hidden_size,
                 device,
                 use_lstm,
                 use_attention):
        super(NER_Model, self).__init__()
        self.voc_size = voc_size
        self.word_emb_size = word_emb_size
        self.char_size = char_size
        self.char_emb_size = char_emb_size
        self.max_word_len = max_word_len
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.device = device
        self.use_lstm = use_lstm
        self.use_attention = use_attention

        # words Embedding
        self.word_embedding = WordEmb(voc_size=self.voc_size,
                                      emb_size=self.word_emb_size,
                                      hidden_size=self.hidden_size,
                                      use_lstm=self.use_lstm,
                                      device=self.device)
        # Char Embedding
        self.char_embedding = CharEmb(char_size=self.char_size,
                                      max_word_size=self.max_word_len,
                                      emb_size=self.char_emb_size,
                                      hidden_size=self.hidden_size,
                                      use_lstm=self.use_lstm,
                                      device=self.device)

        # Classification
        self.classifier = Classifier(input_dim=self.hidden_size,
                                     n_class=2)

        if self.use_attention:
            self.encoder1 = Encoder(input_dim=self.hidden_size,
                                    max_len=self.max_seq_len, device=self.device, num_head=1)

    def to_device(self, device):
        if device == torch.device("cuda"):
            self.word_embedding.words_emb.weight.data.cuda()
            self.char_embedding.char_emb.weight.data.cuda()
            # if self.use_attention:
            #  self.encoder.positionencoder.pe = self.encoder.positionencoder.pe.to(device)

        else:
            self.word_embedding.words_emb.cpu()
            self.char_embedding.char_emb.cpu()
            # if self.use_attention:
            #  self.encoder.positionencoder.pe.cpu()

    def forward(self, words, chars):
        """
        Args:
          1. words: (batch_size,seq_len,idx)
          2. chars: (batch_size,seq_len,max_char_size)

        Return:
          Prediction  [0,1]
        """
        words_embs = self.word_embedding(words)
        chars_embs = self.char_embedding(chars)
        global_embs = words_embs + chars_embs  # (batch_size,seq_len,hidden_size)
        if self.use_attention:
            global_embs = self.encoder1(global_embs)

        preds = self.classifier(global_embs)
        return preds

    def loss(self, words, chars, labels):
        words_embs = self.word_embedding(words)
        chars_embs = self.char_embedding(chars)
        global_embs = words_embs + chars_embs  # (batch_size,seq_len,hidden_size)
        if self.use_attention:
            global_embs = self.encoder1(global_embs)
        loss = self.classifier.loss(global_embs, labels)
        return loss

    def get_representation(self, words, chars):
        with torch.no_grad():
            words_embs = self.word_embedding(words)
            chars_embs = self.char_embedding(chars)
            global_embs = words_embs + chars_embs  # (batch_size,seq_len,2*hidden_size)
            if self.use_attention:
                global_embs = self.encoder1(global_embs)
        return global_embs.detach().cpu().numpy()

    def predict(self, words, chars):
        with torch.no_grad():
            preds = self.forward(words, chars)
            preds = self.classifier.last(preds)
            predictions = torch.round(preds).squeeze()
        return predictions


class ConfusionMatrix:
    def __init__(self, num_class=2):
        self.mat = np.zeros(shape=(num_class, num_class))
        self.num_class = num_class

    def addEstimate(self, trues, preds):
        if type(trues) == np.ndarray:
            assert trues.shape == preds.shape
            for i in range(trues.shape[0]):
                self.mat[trues[i], preds[i]] += 1
        else:
            self.mat[trues, preds] += 1

    def getTP(self):
        if self.num_class == 2:
            return self.mat[1, 1]
        else:
            raise NotImplemented

    def getTN(self):
        if self.num_class == 2:
            return self.mat[0, 0]
        else:
            raise NotImplemented

    def getFN(self):
        if self.num_class == 2:
            return self.mat[1, 0]
        else:
            raise NotImplemented

    def getFP(self):
        if self.num_class == 2:
            return self.mat[0, 1]
        else:
            raise NotImplemented

    def getRecall(self):
        if self.num_class == 2:
            return self.getTP() / (self.getTP() + self.getFN() + 1e-6)
        else:
            raise NotImplemented

    def getPrecision(self):
        if self.num_class == 2:
            return self.getTP() / (self.getTP() + self.getFP() + 1e-6)
        else:
            raise NotImplemented

    def getFscore(self):
        if self.num_class == 2:
            return (2 * self.getRecall() * self.getPrecision()) / (self.getRecall() + self.getPrecision() + 1e-6)
        else:
            raise NotImplemented