from .matCalc import *
from .variables import *
from .mathFunc import *

def forwardProp(lDataset, verify, exp_size):
    # Layer 0
    prod = matrixMulti(lDataset, weights[0])
    tBias = [biases[0] for _ in range(exp_size)]
    sums[0] = matrixAdd(prod, tBias)
    neurons[0] = relu2d(sums[0])
    # Layer 1
    prod = matrixMulti(neurons[0], weights[1])
    tBias = [biases[1] for _ in range(exp_size)]
    sums[1] = matrixAdd(prod, tBias)
    neurons[1] = relu2d(sums[1])
    # Layer 2 (output)
    prod = matrixMulti(neurons[1], weights[2])
    tBias = [biases[2] for _ in range(exp_size)]
    sums[2] = matrixAdd(prod, tBias)
    neurons[2] = softmax(sums[2])
    # Is this loss?
    losses = nllLoss(neurons[2], verify)
    loss = sum(losses) / exp_size
    return loss
