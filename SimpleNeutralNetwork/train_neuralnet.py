import numpy as np
# import sys, os
# sys.path.append(os.pardir)
from NeturalNetwork.TwoLayerNet import *
from part1.part1_in.dataset.mnist import *
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []


train_acc_list = []
test_acc_list = []


iters_num = 10000
train_size = x_train.shape[0] # 60000 numbers

# print(train_size)
batch_size = 100
learning_rate = 0.1

epoch = max(train_size / batch_size, 1) # average repeat times

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

batch_mask = 0
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # from 60000 numbers select 100
    x_batch = x_train[batch_mask]
    # print(x_batch.size) 100 * 784
    t_batch = t_train[batch_mask]
    # print(t_batch.shape) # 100 * 10

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

plt.plot(np.arange(iters_num), train_loss_list)
plt.show()

plt.plot(np.arange(len(train_acc_list)), train_acc_list, label='train acc')
plt.plot(np.arange(len(test_acc_list)), test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right') # legend
plt.show()




