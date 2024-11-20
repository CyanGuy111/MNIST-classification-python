import math
from .variables import *

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

def relu2Derivative(matA):
    h, w = len(matA), len(matA[0])
    matB = [[0 for _ in range(w)] for _ in range(h)]
    for i in range(h):
        matB[i] = reluDerivative(matA[i])
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
    matC = [[0] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if (matA[i][j] == 0):
                matC[i][0] += matB[i][j] * 50
            else:
                matC[i][0] -= matB[i][j] * math.log(matA[i][j])
    return matC

def lossDerivative(matA, matB):
    h = len(matA)
    w = len(matA[0])
    matC = [[0 for _ in range(w)] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            matC[i][j] = matA[i][j] - matB[i][j]
    return matC

def matrixMulti(matA, matB):
    h_A, w_A = len(matA), len(matA[0])
    h_B, w_B = len(matB), len(matB[0])
    if(w_A != h_B):
        raise ValueError('You are multiplying [{},{}] and [{},{}]'.format(h_A, w_A, h_B, w_B))
    matC = [[0 for i in range(w_B)] for j in range(h_A)]
    for i in range(h_A):
        for j in range(w_B):
            for k in range(w_A):
                matC[i][j] += matA[i][k] * matB[k][j]
    return matC

def hadamardProd(matA, matB):
    h_A, w_A = len(matA), len(matA[0])
    h_B, w_B = len(matB), len(matB[0])
    if(h_A != h_B or w_A != w_B):
        raise ValueError('You are adding [{},{}] and [{},{}]'.format(h_A, w_A, h_B, w_B))
    matC = [[0 for i in range(w_A)] for j in range(h_A)]
    for i in range(h_A):
        for j in range(w_A):
            matC[i][j] = matA[i][j] * matB[i][j]
    return matC

def matrixAdd(matA, matB):
    h_A, w_A = len(matA), len(matA[0])
    h_B, w_B = len(matB), len(matB[0])
    if(h_A != h_B or w_A != w_B):
        raise ValueError('You are adding [{},{}] and [{},{}]'.format(h_A, w_A, h_B, w_B))
    matC = [[0 for i in range(w_A)] for j in range(h_A)]
    for i in range(h_A):
        for j in range(w_A):
            matC[i][j] = matA[i][j] + matB[i][j]
    return matC

def vectorMulti(vecB, vecA):
    h, w = len(vecA), len(vecB)
    matC = [[0 for _ in range(w)] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            matC[i][j] = vecA[i] * vecB[j]
    return matC

def vecTranspose(matA):
    matB = [[matA[i]] for i in range(len(matA))]
    return matB

def revecTranspose(matA):
    matB = [matA[i][0] for i in range(len(matA))]
    return matB

def matTranspose(matA):
    matB = [[matA[j][i] for j in range(len(matA))] for i in range(len(matA[0]))]
    return matB

def toRow(matA):
    w = len(matA[0])
    h = len(matA)
    matB = [0 for _ in range(w)]
    for i in range(h):
        for j in range(w):
            matB[j] += matA[i][j]
    return matB

def normalize(a):
    if (a > norm):
        return norm
    return a