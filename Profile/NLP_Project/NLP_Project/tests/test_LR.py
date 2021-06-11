import mylib
import numpy as np
from random import sample

train_x = np.load("../data/eng.train.representation.npy")
train_y = np.load('../data/eng.train.true_labels.npy')

testa_x= np.load("../data/eng.testa.representation.npy")
testa_y = np.load('../data/eng.testa.true_labels.npy')

testb_x= np.load("../data/eng.testb.representation.npy")
testb_y = np.load('../data/eng.testb.true_labels.npy')

train_x = np.array(train_x)
train_y_ = np.zeros(shape=(train_x.shape[0],))
for i in range(train_x.shape[0]):
    if train_y[i] == 'I-PER':
        train_y_[i] = 1
    else:
        train_y_[i] = 0

testa_x = np.array(testa_x)
testa_y_ = np.zeros(shape=(testa_x.shape[0],))
for i in range(testa_x.shape[0]):
    if testa_y[i] == 'I-PER':
        testa_y_[i] = 1
    else:
        testa_y_[i] = 0

testb_x = np.array(testb_x)
testb_y_ = np.zeros(shape=(testb_x.shape[0],))
for i in range(testb_x.shape[0]):
    if testb_y[i] == 'I-PER':
        testb_y_[i] = 1
    else:
        testb_y_[i] = 0


LR = mylib.LogisticRegression(100,0.001,0.0)
LR.fit(train_x,train_y_)
print("###### Trained #####")


mata = mylib.ConfusionMatrix()
er = LR.evaluate(testa_x,testa_y_,mata)
print("#### Test a #####")
mata.PrintEvaluation()

matb = mylib.ConfusionMatrix()
er = LR.evaluate(testb_x,testb_y_,matb)
print("#### Test b #####")

matb.PrintEvaluation()

#print("error_rate: ",er)
