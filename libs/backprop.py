import math
from .matCalc import *
from .mathFunc import *
from .variables import *

def backprop(lDataset, verify):
    cWeight = [[[0 for _ in range(layerDim[i + 1])] for _ in range(layerDim[i])] for i in range(layer_count)]
    cBias = [[0 for _ in range(layerDim[i + 1])] for i in range(layer_count)]
    dWeight = [[[] for _ in range(train_size)] for _ in range(layer_count)]
    dBias = [[[] for _ in range(train_size)] for _ in range(layer_count)]
    dsum = [[[] for _ in range(train_size)] for _ in range(layer_count)]

    for i in range(train_size):
        # Layer 2 
        dsum[2][i] = lossDerivative(neurons[2][i], verify[i])
        dWeight[2][i] = vectorMulti(dsum[2][i], neurons[1][i])
        dBias[2][i] = dsum[2][i]
        # Layer 1
        dsum[1][i] = reluDerivative(matTranspose(matrixMulti(weights[2], vecTranspose(dsum[2][i]))))
        dWeight[1][i] = vectorMulti(dsum[1][i], neurons[0][i])
        dBias[1][i] = dsum[1][i]
        # Layer 0
        dsum[0][i] = reluDerivative(matTranspose(matrixMulti(weights[1], vecTranspose(dsum[1][i]))))
        dWeight[0][i] = vectorMulti(dsum[0][i], lDataset[i])
        dBias[0][i] = dsum[0][i]

    for i in range(layer_count):
        for j in range(train_size):
            for k in range(layerDim[i + 1]):
                for l in range(layerDim[i]):
                    cWeight[i][l][k] += dWeight[i][j][l][k]
                cBias[i][k] += dBias[i][j][k]

    for i in range(layer_count):
        for j in range(layerDim[i + 1]):
            for k in range(layerDim[i]):
                weights[i][k][j] -= (cWeight[i][k][j] * learning_rate) / train_size
            biases[i][j] -= (cBias[i][j] * learning_rate) / train_size