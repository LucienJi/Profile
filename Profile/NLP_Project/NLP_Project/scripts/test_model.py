from NER_Model.DataSet import read_examples_from_file
from NER_Model.NerModel import Ner_Labeler
from NER_Model.NerModel import NER_Model
from NER_Model.DataSet import Dataset
import torch
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter('../data/log')
data_dir = "../data"
exampleTrain = read_examples_from_file(data_dir=data_dir,mode="eng.train")
exampleTesta = read_examples_from_file(data_dir=data_dir,mode="eng.testa")
exampleTestb = read_examples_from_file(data_dir=data_dir,mode="eng.testb")

train_Data = Dataset(exampleTrain,binary=True,f=10)

device = torch.device("cpu")
voc_size = len(train_Data.v2i)
char_size = len(train_Data.c2i)
max_word_len = train_Data.WORD_LEN_CUT
max_seq_len = train_Data.MAX_SENTENCE_LEN_CUT
batch_size = 16
path =  "../data/Model"


test_model = NER_Model(voc_size=voc_size,
                  word_emb_size=300,
                  char_size = char_size,
                  char_emb_size = 16,
                  max_word_len=max_word_len,
                  max_seq_len = max_seq_len,
                  hidden_size = 300,
                  device = device,
                  use_lstm = True,
                  use_attention = True).to(device)

print(test_model)
ner_labler = Ner_Labeler(test_model,train_Data)

"""
w,c,l = train_Data.sample(1)
words = torch.LongTensor(w)
chars = torch.LongTensor(c)
with SummaryWriter(comment='Model') as w:
    w.add_graph(test_model, (words,chars))

"""
loss_list,mat_list = ner_labler.train(1,batch_size,path,augment=True,device=device,test_example=exampleTesta)
repres,words,labels = ner_labler.representation(exampleTrain)
np.save("../data/Repres",repres)
np.save("../data/Words",words)
np.save("../data/Labels",labels)

print("Finished")
