from .variables import *
from .fileReading import *
from .mathFunc import *
from .forward import *
from .backprop import *

def training(current):
    for i in range(train_num):     
        lDataset = []
        verify = [[0 for _ in range(10)] for _ in range(train_size)]
        for j in range(train_size):
            test = dataset[i * train_size + j][1 : 785]
            verify[j][int(dataset[i * train_size + j][0])] = 1
            lDataset.append(test)

        loss = forwardProp(lDataset, verify, train_size)
        backprop(lDataset, verify, train_size)

        if i % 10 == 0:
            print("Epoch: {}. Training progress: [{}/{}] ({:.0f}%); Avg loss: {}".format(current + 1, i + 1, train_num, ((i + 1) / train_num) * 100, loss))
        if i % 100 == 0:
            saveState()
