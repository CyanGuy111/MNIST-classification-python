from .variables import *
from .mathFunc import *

def forwardProp(lDataset, verify, exp_size):
    #input : bsize * 784
    #w0 : 784 * 32
    #b0 : bsize * 32
    #n0 : bsize * 32
    #w1 : 32 * 32
    #b1 : bsize * 32
    #n1 : bsize * 32
    #w2 : 32 * 10
    #b2 : bsize * 10
    #n2 : bsize * 10
    #layer 0
    te = [biases[0] for _ in range(exp_size)]
    sums[0] = matrixAdd(matrixMulti(lDataset, weights[0]), te)
    neurons[0] = relu2d(sums[0])
    #layer 1
    te = [biases[1] for _ in range(exp_size)]
    sums[1] = matrixAdd(matrixMulti(neurons[0], weights[1]), te)
    neurons[1] = relu2d(sums[1])
    #result
    te = [biases[2] for _ in range(exp_size)]
    sums[2] = matrixAdd(matrixMulti(neurons[1], weights[2]), te)
    neurons[2] = softmax(sums[2])

    lossVec = nllLoss(neurons[2], verify)

    s = 0
    for v in lossVec:
        s += v[0]

    return s / exp_size
