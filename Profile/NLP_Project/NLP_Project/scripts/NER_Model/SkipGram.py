import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
"""
SkipGram deprecated
"""

class Word2Vec:
    def __init__(self,
                 dictionary,
                 output_file_name,
                 emb_dimension=300,
                 initial_lr=0.001):
        self.dictionary = dictionary
        self.output_file_name = output_file_name
        self.voc_size = len(dictionary)
        self.emb_dimension = emb_dimension
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.voc_size, self.emb_dimension)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.skip_gram_model.cuda()
        self.optimizer = torch.optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)
        self.batch_size = 256

    def train(self, examples, window_size, epochs):
        process_bar = tqdm(range(int(epochs)))
        for i in process_bar:
            pos_u, pos_v, neg_v = self.dictionary.get_batch_pairs(examples,
                                                                  window_size,
                                                                  self.use_cuda)
            n_batch = int(pos_u.shape[0] / self.batch_size)
            max_size = pos_u.shape[0]
            for i in range(n_batch):
                self.optimizer.zero_grad()
                loss = self.skip_gram_model.forward(pos_u[i:min((i + 1) * self.batch_size, max_size)],
                                                    pos_v[i:min((i + 1) * self.batch_size, max_size)],
                                                    neg_v[i:min((i + 1) * self.batch_size, max_size)])
                loss.backward()
                self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" % (loss, self.optimizer.param_groups[0]['lr']))

            lr = self.initial_lr * (1.0 - 1.0 * i / epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.skip_gram_model.save_embedding(
            self.dictionary.idx_2_word, self.output_file_name, self.use_cuda)

    def get_emb(self, words, labels):
        words_idx, tag_idx = self.dictionary.prepare_data(words, labels)
        emb = self.skip_gram_model.u_embeddings(words_idx)
        return emb, tag_idx

class Dictionary_w2v:
    def __init__(self):
        self.word_2_idx = {}
        self.idx_2_word = {}
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.tag_to_idx = {"I-PER": 1, "O": 0, "I-LOC": 0, 'B-LOC': 0, "I-MISC": 0, 'I-ORG': 0, 'B-ORG': 0, 'B-MISC': 0,
                           self.START_TAG: 2, self.STOP_TAG: 3}
        self.train = []
        self.test = []
        self.id = 0
        self.stemmer = nltk.SnowballStemmer('english')
        self.word_frequency = {}

    def __len__(self):
        return len(self.word_2_idx)

    def read_examples_from_file(self, data_dir, mode="eng.testa"):
        """Creating InputExamples out of a file"""
        file_path = os.path.join(data_dir, "{}".format(mode))
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
        return examples

    def _tokenize(self, words, labels):
        # keep non numeric, stemmer
        last_w = ""
        new_w = []
        new_l = []
        for i in range(len(words)):
            w = words[i]
            l = labels[i]
            if len(re.findall(r'\d', w)) == 0:
                new_w.append(self.stemmer.stem(w))
                new_l.append(l)
                last_w = w
            else:
                if last_w == "<NUM>":
                    continue
                else:
                    new_w.append("<NUM>")
                    new_l.append(l)
                    last_w = "<NUM>"
        return new_w, new_l

    def _prepare_sequence(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def construct_dict(self, path):

        examples = self.read_examples_from_file(path)
        ## examples = list(Input_example)
        num_sentence = len(examples)
        self.sentence_length = 0
        for i in range(len(examples)):
            # words = examples[i].words
            # labels = examples[i].labels
            words, labels = self._tokenize(examples[i].words, examples[i].labels)
            self.sentence_length += len(words)
            for word in words:
                if word not in self.word_2_idx:
                    self.word_2_idx[word] = self.id
                    self.idx_2_word[self.id] = word
                    self.id += 1
                try:
                    self.word_frequency[word] += 1
                except:
                    self.word_frequency[word] = 1
        self.init_sample_table()

    def init_sample_table(self):
        self.sample_table = []
        sample_table_sz = 1e8
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow  # np.array
        count = np.round(ratio * sample_table_sz)
        for id, c in enumerate(count):
            self.sample_table += [id] * int(c)
        self.sample_table = np.array(self.sample_table)

    def get_batch_pairs(self, examples, window_size, use_cuda):
        pos_u = []
        pos_v = []
        for i in range(len(examples)):
            words, lables = examples[i].words, examples[i].labels
            words, lables = self._tokenize(words, lables)
            words_idx = [self.word_2_idx[w] for w in words]
            labels_idx = [self.tag_to_idx[l] for l in lables]
            sz = len(words_idx)
            for i, center_id in enumerate(words_idx):
                for j, neigbor_id in enumerate(words_idx[max(i - window_size, 0):min(sz, i + window_size)]):
                    if i == j:
                        continue
                    pos_u.append(center_id)
                    pos_v.append(neigbor_id)
        neg_v = self.get_neg_v_sampling(pos_v, 5)
        pos_u = torch.autograd.Variable(torch.LongTensor(pos_u))
        pos_v = torch.autograd.Variable(torch.LongTensor(pos_v))
        neg_v = torch.autograd.Variable(torch.LongTensor(neg_v))
        if use_cuda:
            pos_u.cuda()
            pos_v.cuda()
            neg_v.cuda()
        return pos_u, pos_v, neg_v

    def get_neg_v_sampling(self, pos_word_pair, count):
        neg_v = np.random.choice(self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    def prepare_data(self, words, labels):
        words, labels = self._tokenize(words, labels)
        words_idx = self._prepare_sequence(words, self.word_2_idx)
        tag_idx = self._prepare_sequence(labels, self.tag_to_idx)
        return words_idx, tag_idx