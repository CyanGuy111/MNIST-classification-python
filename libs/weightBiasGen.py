import random
import csv
from .variables import *
from .dirName import *

rand_seed = 177014
random.seed = rand_seed

def weightBiasGen():

    for i in range(layer_count):
        weight = [[0 for _ in range(layerDim[i + 1])] for _ in range(layerDim[i])]
        for j in range(layerDim[i]):
            weight[j] = [random.uniform(-2, 2) for _ in range(layerDim[i + 1])]
        biases[i] = [random.uniform(-2, 2) for _ in range(layerDim[i + 1])]
        weights[i] = weight

    csv_file = open(getDir('weightBias/weightBias.csv'), 'w', newline='')

    csvwriter = csv.writer(csv_file)

    for i in range(layer_count):
        csvwriter.writerows(weights[i])
        csvwriter.writerow(biases[i])

    csv_file.close

if(is_random == True):
    weightBiasGen()