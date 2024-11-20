import math
from .matCalc import *

def relu(matA):
    w = len(matA)
    matB = [0 for _ in range(w)]
    for i in range(w):
        matB[i] = max(matA[i], 0)
    return matB

def relu2d(matA):
    h, w = len(matA), len(matA[0])
    matB = [[0 for _ in range(w)] for _ in range(h)]
    for i in range(h):
        matB[i] = relu(matA[i])
    return matB

def reluDerivative(matA):
    w = len(matA)
    matB = [0 for _ in range(w)]
    for j in range(w):
        matB[j] = 1 if matA[j] > 0 else 0
    return matB

def softmax(matA):
    h, w = len(matA), len(matA[0])
    matB = [[0 for _ in range(w)] for _ in range(h)]
    for i in range(h):
        temp = [0 for i in range(w)]
        d = max(matA[i])
        for j in range(w):
            temp[j] = math.exp(matA[i][j] - d)
        s = sum(temp)
        for j in range(w):
            temp[j] = temp[j] / s
        matB[i] = temp
    return matB

# is this loss?
def nllLoss(matA, matB):
    h, w = len(matA), len(matA[0])
    matC = [0 for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if (matA[i][j] == 0):
                matC[i] += matB[i][j] * 50
            else:
                matC[i] -= matB[i][j] * math.log(matA[i][j])
    return matC

def lossDerivative(matA, matB):
    h = len(matA)
    matC = [0 for _ in range(h)]
    for i in range(h):
        matC[i] = matA[i] - matB[i]
    return matC
