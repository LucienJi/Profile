import os
from random import sample
import numpy as np
import itertools
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import nltk
import re


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode="eng.testa"):
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


class Dataset:
    '''
    Static member
    '''
    WORD_LEN_CUT = 22         # The training dataset does not consider words that are longer than WORD_LEN_CUT
    MIN_SENTENCE_LEN_CUT = 4  # Only sentences with the number of words >= MIN_SENTENCE_LEN_CUT [containing *,#] are considered in training data.
    MAX_SENTENCE_LEN_CUT = 65 # Maximum single-sentence input length

    stemmer = nltk.SnowballStemmer('english')
    pattern = re.compile('[a-z]+')
    patternNum = re.compile('[0-9]+')

    '''
    Pre processing functions
    '''
    # Get words,characters,labels
    # in raw form with padding , ready to be converted in to numpy
    def getOriginalData(example):
        words = []
        characters = []
        labels = []

        SENTENCE_LEN_CNT = []  # Count the length distribution in the sentences of the original dataset
        WORD_LEN_CNT = []      # Statistical distribution of the length of words in the original dataset after Stem and number omission as %

        for idx in range(len(example)):
            # Process every sentence
            tmpXWord = ['*']
            tmpXCharacter = [['*']]
            tmpY = ['O']
            SENTENCE_LEN_CNT.append(len(example[idx].words))

            for i in range(len(example[idx].words)):
                StemWord = Dataset.stemmer.stem(example[idx].words[i])
                match = Dataset.pattern.findall(StemWord)
                
                # Replace the number with % 
                StemWord = re.sub(Dataset.patternNum,"%",StemWord)

                WORD_LEN_CNT.append(len(StemWord))

                if len(StemWord) > Dataset.WORD_LEN_CUT:
                    continue

                # If the special characters contain letters, they are retained. Ex. %-year is retained here
                if match:
                    tmpXWord.append(StemWord)
                    tmpXCharacter.append(list(StemWord))
                    tmpY.append(example[idx].labels[i])
                    continue
                
                # If it only contains  special characters
                # If it's ? or ! it's changed uniformly to '.'
                if StemWord in ["?","!"]:
                    StemWord = "."
                # If it's a ',' or a '.' or a simple number (%), it is retained
                if StemWord in [",",".","%"]:
                    tmpXWord.append(StemWord)
                    tmpXCharacter.append(list(StemWord))
                    tmpY.append(example[idx].labels[i])
                
            tmpXWord += ['#']
            tmpXCharacter.append(['#'])
            tmpY += ['O']

            #  len(tmpXWord) is truncated in to pieces of length (Dataset.MAX_SENTENCE_LEN_CUT)
            if len(tmpXWord) >= Dataset.MIN_SENTENCE_LEN_CUT:
                start = 0
                end = Dataset.MAX_SENTENCE_LEN_CUT

                assert len(tmpXWord[start:end])==len(tmpXCharacter[start:end]),"LENGTH ERROR"
                assert len(tmpY[start:end])==len(tmpXCharacter[start:end]),"LENGTH ERROR"

                words.append(tmpXWord[start:end])
                characters.append(tmpXCharacter[start:end])
                labels.append(tmpY[start:end])

                while (len(tmpXWord)-end)>(Dataset.MIN_SENTENCE_LEN_CUT-1):
                    assert len(tmpXWord[start:end])==len(tmpXCharacter[start:end]),"LENGTH ERROR"
                    assert len(tmpY[start:end])==len(tmpXCharacter[start:end]),"LENGTH ERROR"
                    
                    start += Dataset.MAX_SENTENCE_LEN_CUT
                    end += Dataset.MAX_SENTENCE_LEN_CUT

                    words.append(tmpXWord[start:end])
                    characters.append(tmpXCharacter[start:end])
                    labels.append(tmpY[start:end])

        originalDataLength = len(words)

        return words,characters,labels,originalDataLength,SENTENCE_LEN_CNT,WORD_LEN_CNT
    
    '''
    Word,Label: singleList2indexList
    Character : chars2indexList
    '''
    def singleList2indexList(singleList,dic,MAX_LEN=-1):
        """
      Args:
        singleList: list of string
        dic: dict[string] = int idx
        MAX_LEN: len(return) < max_len

      Return :list of idx
        """
        if MAX_LEN==-1:
            return [dic[w] for w in singleList]
        else:
            tmp = [0 for _ in range(MAX_LEN)]
            assert len(singleList) <= MAX_LEN, f"{len(singleList)},{MAX_LEN}"
            for i in range(len(singleList)):
                tmp[i] = dic[singleList[i]]
            return tmp

    def indexList2singleList(indexList,dic):
        return [dic[idx] for idx in indexList]

    def showSingleList(indexList,dic):
        print(" ".join(Dataset.indexList2singleList(indexList,dic)))

    def chars2indexList(wordList,c2i,MAX_WORD_LEN,MAX_SENTENCE_LEN):
        """
      Args: 
        wordList: list of string == sentence after tokenizing and padding
        MAX_WORD_LEN: len of word < 

        """
        res = [[0 for _ in range(MAX_WORD_LEN)] for _ in range(MAX_SENTENCE_LEN)]
        for i,word in enumerate(wordList):
            for j,c in enumerate(word):
                res[i][j] = c2i[c]
        return res

    def indexList2chars(indexList,i2c):
        MAX_SENTENCE_LEN = len(indexList)
        MAX_WORD_LEN = len(indexList[0])
        res = [[0 for _ in range(MAX_WORD_LEN)] for _ in range(MAX_SENTENCE_LEN)]
        for i,word in enumerate(indexList):
            for j,c in enumerate(word):
                res[i][j] = i2c[c]
        return res

    def showCharList(indexList,i2c):
        tmp = Dataset.indexList2chars(indexList,i2c)
        res = ""
        for word in tmp:
            res += "".join(word)
            res += " "
        print(res)

    '''
    Augmentation
    '''
    # If the sentence length exceeds MAX_LEN, no augmentaion is done
    # 1 sentence containing a person's name is expanded into a number of 'factor' sentences
    def augment(sentence,labelList,name_dict,factor=10,MAX_LEN=20):
        if ('I-PER' not in labelList and 'B-PER' not in labelList) or len(sentence) > MAX_LEN:
            return None
        else:
            wordNum = len(sentence)
            res  = [[0 for _ in range(wordNum)] for _ in range(factor)]
            # [l,r[ is the person's name
            l = 0 
            r = 0
            while r<wordNum:
                if 'PER' in labelList[l]:
                    while (r<wordNum) and ( 'PER' in labelList[r]):
                        r+=1
                    # 得到一个name[l,r[，此时 要么r已经出界，要么r停在一个不是人名的位置
                    nameLen = r-l # length of the name
                    # Sample a name with the same ke
                    tmpId = np.random.choice(len(name_dict[nameLen]),factor)
                    assert len(tmpId)==factor, "ERROR"

                    for i,id in enumerate(tmpId):
                        for j,name in enumerate(name_dict[nameLen][id]):
                            res[i][l+j] = name
                    
                    l=r
                else:
                    for i in range(factor):
                        res[i][l] = sentence[l]
                    l+=1
                    r+=1
            return res


    # Construct Various Dict
    def getDict(self):
        all_words = list(itertools.chain(*self.words))
        vocab, v_count = np.unique(all_words, return_counts=True) # 统计每一个词语出现的次数
        vocab = vocab[np.argsort(v_count)[::-1]]                  # vocab按出现次数从大到小排序,最多出现的index是0
        v_count = np.sort(v_count)[::-1]                          # 从大到小排列v_count，与vocab对齐

        chars = set(itertools.chain(*list(itertools.chain(*self.characters))))

        # 添加UNKOWN的符号 ?
        vocab = np.append(vocab,"?")
        v_count = np.append(v_count,0)
        chars.add("?")

        # Vocabulary
        v2i = {v: (i+1) for i, v in enumerate(vocab)}
        v2i[""] = 0                                              # 0用于padding
        i2v = {i: v for v, i in v2i.items()}

        # Character
        c2i = {c: (i+1) for i,c in enumerate(chars)}
        c2i[""] = 0                                              # 0用于padding
        i2c = {i: c for c,i in c2i.items()}

        # Labels
        # all_labels = set(itertools.chain(*labels))
        all_labels = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

        # i2l, l2i
        if self.binary:
            l2i = {}
            i2l = {0:'O',1:'I-PER'} # 其他都记为 'O'
            for l in all_labels:
                if 'PER' in l:
                    l2i[l] = 1
                else:
                    l2i[l] = 0
        else:
            l2i = {l: i for i, l in enumerate(all_labels)}
            i2l = {i: l for l, i in l2i.items()}
        return vocab,v2i,i2v,c2i,i2c,l2i,i2l
    
    # Construct Name Dict
    def getNameDict(self):
        name_dict = {}
        for i in range(len(self.words)):
            j = 0 
            while j<len(self.words[i]):
                if "PER" in self.labels[i][j]:
                    tmpName = []
                    while (j<len(self.words[i])) and "PER" in self.labels[i][j]:
                        tmpName.append(self.words[i][j])
                        j+=1
                    # 得到一个name，此时 要么j已经出界，要么j停在一个不是人名的位置
                    if len(tmpName) in name_dict.keys():
                        name_dict[len(tmpName)].append(tmpName)
                    else:
                        name_dict[len(tmpName)]=[]
                        name_dict[len(tmpName)].append(tmpName)
                j+=1    
        return name_dict
    
    def getTrainData(self):
        xWords = []         # SampleSize,MAX_SENTENCE_LEN
        xCharacters = []    # SampleSize,MAX_SENTENCE_LEN,MAX_WORD_LEN
        yLabel = []         # SampleSize,MAX_SENTENCE_LEN

        for i in range(0,len(self.words)):
            xWords.append(Dataset.singleList2indexList(self.words[i],self.v2i,Dataset.MAX_SENTENCE_LEN_CUT))
            xCharacters.append(Dataset.chars2indexList(self.characters[i],self.c2i,Dataset.WORD_LEN_CUT,Dataset.MAX_SENTENCE_LEN_CUT))
            yLabel.append(Dataset.singleList2indexList(self.labels[i],self.l2i,Dataset.MAX_SENTENCE_LEN_CUT))

        xWords = np.array(xWords)
        xCharacters = np.array(xCharacters)
        yLabel = np.array(yLabel)

        return xWords,xCharacters,yLabel
    
    # Judge if after the transformation a word is in the vocab
    def wordInDict(self,word):
        StemWord = Dataset.stemmer.stem(word)
        match = Dataset.pattern.findall(StemWord)
        StemWord = re.sub(Dataset.patternNum,"%",StemWord)
        # 如果是 ? 或者 ! 统一变为 .
        if StemWord in ["?","!"]:
            StemWord = "."
        
        if len(StemWord)>Dataset.WORD_LEN_CUT:
            StemWord = StemWord[0:Dataset.WORD_LEN_CUT]
        
        return StemWord in self.v2i.keys()
        

    '''
    Print Functions
    '''
    def showLengthDistribution(self):
        if len(self.SENTENCE_LEN_CNT)==0 or len(self.WORD_LEN_CNT)==0:
            print("Please call getOriginalData")
            return
        plt.figure(figsize=(10,6))
        plt.hist(self.SENTENCE_LEN_CNT,bins=max(self.SENTENCE_LEN_CNT),density=True)
        plt.vlines(Dataset.MAX_SENTENCE_LEN_CUT,0,0.1,label = "Max Sentence Length",colors="r")
        plt.vlines(Dataset.MIN_SENTENCE_LEN_CUT,0,0.1,label = "Min Sentence Length",colors="b")
        plt.xlabel("Length")
        plt.ylabel("Density")
        plt.title("Sentence length distribution")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10,6))
        plt.hist(self.WORD_LEN_CNT,bins=max(self.WORD_LEN_CNT),density=True)
        plt.vlines(Dataset.WORD_LEN_CUT,0,0.1,label = "Max Word Length",colors="r")
        plt.xlabel("Length")
        plt.ylabel("Density")
        plt.title("Word length distribution")
        plt.legend()
        plt.show()

    def showLableDistribution(labelList,msg,f):
        tmp = {}
        for i in range(len(labelList)):
            for label in labelList[i]:
                if label in tmp.keys():
                    tmp[label]+=1
                else:
                    tmp[label]=1

        cnt = 0
        for key,value in tmp.items():
            print(key,"\t",value)
            cnt+=value

        tmp = sorted(tmp.items(), key=lambda item:item[1], reverse=True)

        plt.bar([d[0] for d in tmp], [d[1]/cnt for d in tmp],width = 0.4)
        plt.ylabel("Percentage")
        plt.xlabel("Labels")
        plt.title(msg+f" factor = {f}")
        plt.show()

    def __init__(self,example,binary=True,f=20,name_dict_Outside=None):
        self.binary = binary
        self.f = f
        self.words,self.characters,self.labels,self.originalDataLength,self.SENTENCE_LEN_CNT,self.WORD_LEN_CNT = Dataset.getOriginalData(example)
        self.vocab,self.v2i,self.i2v,self.c2i,self.i2c,self.l2i,self.i2l = self.getDict()
        self.name_dict = self.getNameDict()
        if name_dict_Outside != None:
            print("Using the name dict given by user")
            # 给定的人名
            self.name_dict = name_dict_Outside

        # Augmenting Original Data

        # Dataset.showLableDistribution(self.labels,"Before Augmentation",self.f)
        print(f"Before augmentaion, num of senteces is {self.originalDataLength}")
        for id in range(self.originalDataLength):
            tmpRes = Dataset.augment(self.words[id],self.labels[id],self.name_dict,factor = self.f)
            if tmpRes !=None:
                self.words += tmpRes
                self.labels += [self.labels[id] for _ in range(f)]
                self.characters += [[list(w) for w in tmpRes[i]] for i in range(f)]
        print(f"After augmentaion, num of senteces is {len(self.words)}")
        # Dataset.showLableDistribution(self.labels,"After Augmentation",self.f)
        
        self.xWords,self.xCharacters,self.yLabel = self.getTrainData()
        self.totalDataLength = len(self.words)

    # 用于sample得到一个Batchsize,augment=False只在[0,originalDataLength[中sample数字
    def sample(self, n, augment = True):
        if not augment:
            b_idx = np.random.randint(0, self.originalDataLength, n)
        else:
            b_idx = np.random.randint(0, self.totalDataLength, n)
        # n,MAX_SENTENCE_LEN | n,MAX_SENTENCE_LEN,MAX_WORD_LEN | n,MAX_SENTENCE_LEN
        return self.xWords[b_idx],self.xCharacters[b_idx],self.yLabel[b_idx]
    
    def sample_test(self,words,chars,labels,raws,max_batchsize = 128):
        w,c,l,raw = words,chars,labels,raws
        assert len(w) ==len(c),"Test Error1"
        assert len(c) == len(l),"Test Error2"
        assert len(l) == len(raw),"Test Error3"

        test_wlist = [] # too many to be put in one CUDA,so we separate the sample
        test_clist = []
        test_llist = []
        test_rlist = []
        beg = 0
        end = max_batchsize
        length = len(w)
        while end < length:
          test_wlist.append(w[beg:end])
          test_clist.append(c[beg:end])
          test_llist.append(l[beg:end])
          test_rlist.append(raw[beg:end])
          beg = end
          end += max_batchsize
        test_wlist.append(w[beg:-1])
        test_clist.append(c[beg:-1])
        test_llist.append(l[beg:-1])
        test_rlist.append(raw[beg:-1])
        return test_wlist,test_clist,test_llist,test_rlist

      
        
    
    # newData : [[句子1单词1,..,句子1单词n],[句子2单词1,..,句子2单词n],...] , (Size,)
    # xWordsNew : (Size,MAX_SENTENCE_LEN)
    # xCharactersNew : (Size,MAX_SENTENCE_LEN,MAX_WORD_LEN)
    # 句子太长切成若干份？
    def eval(self,newData):
        size = len(newData)
        xWordsNew = []
        xCharactersNew = []

        stemmer = nltk.SnowballStemmer('english')
        pattern = re.compile('[a-z]+')              # 找到字符串中是否含有小写字母
        patternNum = re.compile('[0-9]+')

        # 要用stemmer 和同样的预处理！切断太长的词 MAX_WORD_LEN
        for idx in range(size):
            tmpXWord = ['*']
            tmpXCharacter = [['*']]
            for i,word in enumerate(newData[idx]):
                StemWord = stemmer.stem(word)
                match = pattern.findall(StemWord)
                # 将数字替换成 % ！ 注意，这里我们原数据集中，没有%
                StemWord = re.sub(patternNum,"%",StemWord)
                # 如果是 ? 或者 ! 统一变为 .
                if StemWord in ["?","!"]:
                    StemWord = "."
    
                if len(StemWord)>Dataset.WORD_LEN_CUT:
                    StemWord = StemWord[0:Dataset.WORD_LEN_CUT]
                # 对词语处理
                if StemWord in self.v2i.keys():
                    tmpXWord.append(StemWord)
                else:
                    tmpXWord.append("?")
                
                tmpChar = []
                for c in list(StemWord):
                    if c in self.c2i.keys():
                        tmpChar.append(c)
                    else:
                        tmpChar.append('?')

                tmpXCharacter.append(tmpChar)
            
            tmpXWord += ['#']
            tmpXCharacter.append(['#'])
            print(tmpXWord)
            print(tmpXCharacter)
            print("======")
            xWordsNew.append(Dataset.singleList2indexList(tmpXWord,self.v2i,Dataset.MAX_SENTENCE_LEN_CUT))
            xCharactersNew.append(Dataset.chars2indexList(tmpXCharacter,self.c2i,Dataset.WORD_LEN_CUT,Dataset.MAX_SENTENCE_LEN_CUT))

        assert len(xWordsNew) == size, "ERROR"
        assert len(xCharactersNew) == size, "ERROR"
        
        return np.array(xWordsNew),np.array(xCharactersNew)
    def transform(self,example):
        size = len(example)
        
        xWordsTest = []
        xCharactersTest = []
        yLabelTest = []
        words_4_repres = []
        # 要用stemmer 和同样的预处理！切断太长的词 MAX_WORD_LEN
        for idx in range(size):
            tmpXWord = ['*']
            tmpXCharacter = [['*']]
            tmpYLabel = ['O']
            
            for i,word in enumerate(example[idx].words):
                StemWord = Dataset.stemmer.stem(word)
                match = Dataset.pattern.findall(StemWord)
                # 将数字替换成 % ！ 注意，这里我们原数据集中，没有%
                StemWord = re.sub(Dataset.patternNum,"%",StemWord)
                # 如果是 ? 或者 ! 统一变为 .
                if StemWord in ["?","!"]:
                    StemWord = "."

                if len(StemWord)>Dataset.WORD_LEN_CUT:
                    StemWord = StemWord[0:Dataset.WORD_LEN_CUT]

                # 对词语处理
                if StemWord in self.v2i.keys():
                    tmpXWord.append(StemWord)
                else:
                    tmpXWord.append("?")
                
                tmpChar = []
                for c in list(StemWord):
                    if c in self.c2i.keys():
                        tmpChar.append(c)
                    else:
                        tmpChar.append('?')
                tmpXCharacter.append(tmpChar)

                tmpYLabel.append(example[idx].labels[i])

            tmpXWord += ['#']
            tmpXCharacter.append(['#'])
            tmpYLabel.append('O')
            
            assert len(tmpXWord)==len(tmpXCharacter), "ERROR"
            assert len(tmpYLabel)==len(tmpXCharacter), "ERROR"
            
            start = 0
            end = Dataset.MAX_SENTENCE_LEN_CUT

            assert len(tmpXWord[start:end])==len(tmpXCharacter[start:end]),"LENGTH ERROR"
            assert len(tmpYLabel[start:end])==len(tmpXCharacter[start:end]),"LENGTH ERROR"

            xWordsTest.append(Dataset.singleList2indexList(tmpXWord[start:end],self.v2i,Dataset.MAX_SENTENCE_LEN_CUT))
            xCharactersTest.append(Dataset.chars2indexList(tmpXCharacter[start:end],self.c2i,Dataset.WORD_LEN_CUT,Dataset.MAX_SENTENCE_LEN_CUT))
            yLabelTest.append(Dataset.singleList2indexList(tmpYLabel[start:end],self.l2i,Dataset.MAX_SENTENCE_LEN_CUT))
            words_4_repres.append(tmpXWord[start:end])
            # 如果还有剩余就继续切分,统计到size里面
            while (len(tmpXWord)-end)>0:
                size += 1
                assert len(tmpXWord[start:end])==len(tmpXCharacter[start:end]),"LENGTH ERROR"
                assert len(tmpYLabel[start:end])==len(tmpXCharacter[start:end]),"LENGTH ERROR"

                start += Dataset.MAX_SENTENCE_LEN_CUT
                end += Dataset.MAX_SENTENCE_LEN_CUT
                words_4_repres.append(tmpXWord[start:end])
                xWordsTest.append(Dataset.singleList2indexList(tmpXWord[start:end],self.v2i,Dataset.MAX_SENTENCE_LEN_CUT))
                xCharactersTest.append(Dataset.chars2indexList(tmpXCharacter[start:end],self.c2i,Dataset.WORD_LEN_CUT,Dataset.MAX_SENTENCE_LEN_CUT))
                yLabelTest.append(Dataset.singleList2indexList(tmpYLabel[start:end],self.l2i,Dataset.MAX_SENTENCE_LEN_CUT))

        assert len(xWordsTest) == size, "ERROR"
        assert len(xCharactersTest) == size, "ERROR"
        assert len(yLabelTest) == size, "ERROR"
        
        return np.array(xWordsTest),np.array(xCharactersTest),np.array(yLabelTest),words_4_repres