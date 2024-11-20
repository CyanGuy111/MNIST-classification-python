from .variables import *
from .mathFunc import *
from .forward import *
from .backprop import *

def testing():
    lDataset = []
    verify = [[0 for _ in range(10)] for _ in range(test_size)]
    ground_truth = []

    for j in range(test_size):
        test = dataset[9000 + j][1 : 785]
        verify[j][int(dataset[9000 + j][0])] = 1
        ground_truth.append(dataset[9000 + j][0])
        lDataset.append(test)

    loss = forwardProp(lDataset, verify, test_size)
    score = 0

    for i in range(10):
        prediction = 0
        maxn = neurons[2][i][0]
        for j in range(10):
            print(neurons[2][i][j])
            if maxn < neurons[2][i][j]:
                prediction = j
                maxn = neurons[2][i][j]
        if prediction == int(ground_truth[i]):
            score += 1
        print(prediction, int(ground_truth[i]))
        print("\n")

    print("Test result: {}/{} ({:.0f}%). Avg loss: {}".format(score, test_size, (score / test_size) * 100, loss))

