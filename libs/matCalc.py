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

def matrixAdd(matA, matB, c = 1):
    h_A, w_A = len(matA), len(matA[0])
    h_B, w_B = len(matB), len(matB[0])
    if(h_A != h_B or w_A != w_B):
        raise ValueError('You are a dumbass Add')
    matC = [[0 for i in range(w_A)] for j in range(h_A)]
    for i in range(h_A):
        for j in range(w_A):
            matC[i][j] = matA[i][j] + c * matB[i][j]
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

def matTranspose(matA):
    matB = [matA[j][i] for j in range(len(matA)) for i in range(len(matA[0]))]
    return matB