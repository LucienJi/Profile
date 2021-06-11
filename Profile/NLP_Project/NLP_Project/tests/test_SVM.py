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


svm = mylib.svm(1,"LinearKernel")
svm.fit(testa_x,testa_y_)
print("###### Trained #####")


mata = mylib.ConfusionMatrix()
er = svm.evaluate(testa_x,testa_y_,mata)
print("#### Test a #####")
mata.PrintEvaluation()

matb = mylib.ConfusionMatrix()
er = svm.evaluate(testb_x,testb_y_,matb)
print("#### Test b #####")

matb.PrintEvaluation()

#print("error_rate: ",er)



"""
def f(x):
    return 1 if x-0.5 > 0 else 0

x = np.random.randn(1000)
y = np.zeros_like(x)
for i in range(x.shape[0]):
    y[i] = f(x[i])

svm = mylib.svm(1,"LinearKernel")

matb = mylib.ConfusionMatrix()
svm.fit(x,y)
er = svm.evaluate(x,y,matb)
print("#### Test b #####")
matb.PrintEvaluation()
"""