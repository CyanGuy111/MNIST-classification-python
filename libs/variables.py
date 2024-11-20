
layer_count = 3
weights = [[] for i in range(layer_count)]
biases = [[] for i in range(layer_count)]
sums = [[] for i in range(layer_count)]
neurons = [[] for i in range(layer_count)]
rawInput = []
layerDim = [784, 32, 32, 10]
dataset = []
train_size = 10 #batch size
test_size = 1000 #test batch size
dataset_size = 10000    
train_num = 1000 #number of iteration in 1 epoch
train_num = int((dataset_size - test_size) / train_size)
is_random = True
learning_rate = 0.05
norm = 10000
epoch = 1