
layer_count = 3
weights = [[] for i in range(layer_count)]
biases = [[] for i in range(layer_count)]
sums = [[] for i in range(layer_count)]
neurons = [[] for i in range(layer_count)]
rawInput = []
layerDim = [784, 16, 16, 10]
dataset = []
train_size = 10
test_size = 1000
dataset_size = 10000
train_num = 200
# train_num = int((dataset_size - test_size) / train_size)
is_random = True
learning_rate = 0.01
epoch = 5