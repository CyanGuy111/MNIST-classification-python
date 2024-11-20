import csv
import random
from .variables import *
from .dirName import *

rand_seed = 177014
random.seed = rand_seed

def strToReal(matA):
    h = len(matA)
    for i in range(h):
        w = len(matA[i])
        for j in range(w):
            matA[i][j] = float(matA[i][j])

def weightBiasRead():
    csv_file = open(getDir('weightBias/weightBias.csv'), 'r')
    reader = csv.reader(csv_file)

    for row in reader:
        rawInput.append(row)

    csv_file.close

    strToReal(rawInput)

    cnt = 0

    for i in range(layer_count):
        weight = [[0 for _ in range(layerDim[i + 1])] for _ in range(layerDim[i])]
        for j in range(layerDim[i]):
            weight[j] = rawInput[cnt]
            cnt += 1
        biases[i] = rawInput[cnt]
        weights[i] = weight
        cnt += 1

def dataRead():
    csv_file = open(getDir('test_set/MNIST.csv'), 'r')
    reader = csv.reader(csv_file)

    # 10k tests
    for row in reader:
        dataset.append(row)
    strToReal(dataset)
    csv_file.close

def saveState():
    csv_file = open(getDir('weightBias/weightBias.csv'), 'w', newline='')

    csvwriter = csv.writer(csv_file)

    for i in range(layer_count):
        csvwriter.writerows(weights[i])
        csvwriter.writerow(biases[i])

    csv_file.close

def weightBiasGen():
    for i in range(layer_count):
        te = [[0 for _ in range(layerDim[i + 1])] for _ in range(layerDim[i])]
        for j in range(layerDim[i]):
            te[j] = [random.uniform(-10, 10) for _ in range(layerDim[i + 1])]
        biases[i] = [random.uniform(-10, 10) for _ in range(layerDim[i + 1])]
        weights[i] = te

    csv_file = open(getDir('weightBias/weightBias.csv'), 'w', newline='')

    csvwriter = csv.writer(csv_file)

    for i in range(layer_count):
        csvwriter.writerows(weights[i])
        csvwriter.writerow(biases[i])

    csv_file.close

if(is_random == True):
    weightBiasGen()

if(is_random == False):
    weightBiasRead()

dataRead()