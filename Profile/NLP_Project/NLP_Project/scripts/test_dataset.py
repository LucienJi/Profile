from NER_Model.DataSet import read_examples_from_file
import matplotlib.pyplot as plt
import itertools

data_dir = "../data"
exampleTrain = read_examples_from_file(data_dir=data_dir,mode="eng.train")
exampleTesta = read_examples_from_file(data_dir=data_dir,mode="eng.testa")
exampleTestb = read_examples_from_file(data_dir=data_dir,mode="eng.testb")


example = exampleTestb

tmpCnt = 0
tmp = {}
for i in range(len(example)):
    for label in example[i].labels:
        tmpCnt +=1
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
plt.show()


example = exampleTrain

categoryWordlist = {} # {'I-PER':[word1,word2,...,]}
word2Category = {}    # {word1: {'I-PER':[article id1,article id2],'I-LOC':[article id1]} }

for id,article in enumerate(example):
    l = len(article.words)
    for i in range(l):
        word = article.words[i]
        category = article.labels[i]
        if category not in categoryWordlist.keys():
            categoryWordlist[category]=[word]
        else:
            categoryWordlist[category] += [word]
        
        if word not in word2Category.keys():
            word2Category[word]= {}
            word2Category[word][category] = [id]
        else:
            if category in word2Category[word].keys():
                word2Category[word][category].append(id)
            else:
                word2Category[word][category] = [id]

with open('../data/Multiple Tags.txt', 'w') as f:
    for word,categoryDict in word2Category.items():
        if len(categoryDict)>1:
            f.write("Word \t Multiple tags\n")
            f.write(f"{word} : {categoryDict.keys()}\n")
            # for word has multiple tags, we show the first sentence in this tags
            for category,ids in categoryDict.items():
                f.write(f"\t {category} in sentence\n")
                f.write("\t "+" ".join(example[ids[0]].words)+"\n")
            f.write(f"----------------------------\n")
# We also regard 's as a case in Special characters
with open('../data/Special characters Tags.txt', 'w') as f:
    for word,categoryDict in word2Category.items():
        if len(categoryDict)>1 and not word.isalpha():
            f.write("Word \t Multiple tags\n")
            f.write(f"{word} : {categoryDict.keys()}\n")
            # for word has multiple tags, we show the first sentence in this tags
            for category,ids in categoryDict.items():
                f.write(f"\t {category} in sentence\n")
                f.write("\t "+" ".join(example[ids[0]].words)+"\n")
            f.write(f"----------------------------\n")
# Check if the same word can have different labels in the same sentence.
with open('../data/Multiple Tags in One Sentence.txt', 'w') as f:
    for word,categoryDict in word2Category.items():
        if len(categoryDict)>1:
            tmpList = [list(set(l)) for l in categoryDict.values()] # Remove the case: same word same sentence same tags
            tmp = list(itertools.chain(*tmpList))
            tmpSet = set()
            ids = []
            for id in tmp:
                if id not in tmpSet:
                    tmpSet.add(id)
                else:
                    ids.append(id)
            # Print the corresponding label,sentence for the word
            if len(ids)>0:
                for id in ids:
                    f.write(f"\"{word}\" has different labels in the same sentence.\n")
                    f.write(f"Tags : \n")
                    for key,value in categoryDict.items():
                        if id in value:
                            f.write("\t "+key+"\n")
                    f.write("Sentence : \n" + " ".join(example[id].words)+"\n")
                    f.write("--------------------\n")
                f.write("=======================\n")