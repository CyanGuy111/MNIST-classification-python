import csv
import random
import time
from libs.variables import *
from libs.matCalc import *
from libs.dirName import *
from libs.fileReading import *
from libs.mathFunc import * 
from libs.weightBiasGen import *
from libs.forward import *
from libs.training import *
from libs.testing import *

for i in range(epoch):
    training(i)
    testing()

