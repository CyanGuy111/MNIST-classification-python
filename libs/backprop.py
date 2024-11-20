import math
from .mathFunc import *
from .variables import *

def backprop(lDataset, verify, exp_size):
    dw = [[],[],[]]
    db = [[],[],[]]
    delta = [[],[],[]]

    #derivative of loss wrt s2
    delta[2] = lossDerivative(neurons[2], verify)
    #derivative of loss wrt s1
    delta[1] = hadamardProd(matrixMulti(delta[2], matTranspose(weights[2])), relu2Derivative(neurons[1]))
    # delta[1] = hadamardProd(delta[2], relu2Derivative(neurons[1]))
    #derivative of loss wrt s0
    delta[0] = hadamardProd(matrixMulti(delta[1], matTranspose(weights[1])), relu2Derivative(neurons[0]))
    # delta[0] = hadamardProd(delta[1], relu2Derivative(neurons[0]))
    #derivative of loss wrt w2
    dw[2] = matrixMulti(matTranspose(neurons[1]), delta[2])
    #derivative of loss wrt b2
    db[2] = toRow(delta[2])
    #derivative of loss wrt w1
    dw[1] = matrixMulti(matTranspose(neurons[0]), delta[1])
    #derivative of loss wrt b1
    db[1] = toRow(delta[1])
    #derivative of loss wrt w0
    dw[0] = matrixMulti(matTranspose(lDataset), delta[0])
    #derivative of loss wrt b0
    db[0] = toRow(delta[0])

    for l in range(3):
        # print(dw[l])
        # print(db[l])
        for i in range(layerDim[l + 1]):
            for j in range(layerDim[l]):
                weights[l][j][i] -= (normalize(dw[l][j][i]) * learning_rate)
            biases[l][i] -= (normalize(db[l][i]) * learning_rate)
