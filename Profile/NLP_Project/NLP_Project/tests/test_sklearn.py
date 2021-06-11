import sklearn.metrics as metrics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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


lr = LogisticRegression()
lr.fit(train_x,train_y_)

pred_a = lr.predict(testa_x)
pred_b = lr.predict(testb_x)
fscore_a = metrics.f1_score(testa_y_,pred_a)
fscore_b = metrics.f1_score(testb_y_,pred_b)
recall_a = metrics.recall_score(testa_y_,pred_a)
recall_b = metrics.recall_score(testb_y_,pred_b)

pre_a = metrics.precision_score(testa_y_,pred_a)
pre_b = metrics.precision_score(testb_y_,pred_b)
print("### LR ###")

print(fscore_a,fscore_b)
print(recall_a,recall_b)
print(pre_a,pre_b)

svm = SVC(kernel='linear',coef0=1.0)
svm.fit(train_x, train_y_)
pred_a = svm.predict(testa_x)
pred_b = svm.predict(testb_x)
fscore_a = metrics.f1_score(testa_y_,pred_a)
fscore_b = metrics.f1_score(testb_y_,pred_b)
recall_a = metrics.recall_score(testa_y_,pred_a)
recall_b = metrics.recall_score(testb_y_,pred_b)

pre_a = metrics.precision_score(testa_y_,pred_a)
pre_b = metrics.precision_score(testb_y_,pred_b)
print("### SVM ###")
print(fscore_a,fscore_b)
print(recall_a,recall_b)
print(pre_a,pre_b)

