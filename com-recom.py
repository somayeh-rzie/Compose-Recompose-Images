import cv2
import os
import numpy as np
import math
import random
import sys
from statistics import mean
import matplotlib.pyplot as plt
import time

BLOCK_DIMENSION = 8

number_of_hidden_cells = 32
number_of_correct_estimations = 0
number_of_networks = 8
number_of_epochs = 60

total_test_psnrs = []
total_train_costs = []

path = os.path.join('shabake asabi')


# Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# separating array into NxN chunks
def ressample(arr, N):
    A = []
    for v in np.vsplit(arr, arr.shape[0] // N):
        A.extend([*np.hsplit(v, arr.shape[0] // N)])
    return np.array(A)


# load dataset
def read_from_file(address):

    path = os.path.join('shabake asabi',address)
    result_set = []

    for file in os.listdir(path):
        # Check whether file is in jpg format or not
        if file.endswith(".jpg"):
            img_path = os.path.join(path,file)
            img = cv2.imread(img_path, 0) # read image as grayscale
            # plt.imshow(img , cmap='gray')
            # plt.show()
            result_set.append(img)
    return result_set


# psnr calculator
def calculate_PSNR(cost):
    a = math.pow(255, 2) * math.pow(BLOCK_DIMENSION, 2)
    a = a/cost
    return 10*(math.log(a, 10))

# print(calculate_PSNR())


def standard_train():

    W1 = []
    W2 = []
    b1 = []
    b2 = []
    learning_rates = []

    for i in range(number_of_networks):
        W1.append(np.random.normal(size=(number_of_hidden_cells, BLOCK_DIMENSION*BLOCK_DIMENSION)))
        W2.append(np.random.normal(size=(BLOCK_DIMENSION*BLOCK_DIMENSION, number_of_hidden_cells)))
        b1.append(np.random.normal(size=(number_of_hidden_cells, 1)))
        b2.append(np.random.normal(size=(BLOCK_DIMENSION*BLOCK_DIMENSION, 1)))
        learning_rates.append(random.uniform(0, 1))

    for epoch in range(number_of_epochs):

        np.random.shuffle(train_set)
            
        for train_data in train_set:

            blocks = ressample(train_data, BLOCK_DIMENSION)

            batches = [blocks[x:(x+(math.floor(1024/number_of_networks)))] for x in range(0, len(blocks), (math.floor(1024/number_of_networks)))]

            i=0

            for batch in batches:

                grad_W1 = np.zeros((number_of_hidden_cells, BLOCK_DIMENSION*BLOCK_DIMENSION))
                grad_W2 = np.zeros((BLOCK_DIMENSION*BLOCK_DIMENSION, number_of_hidden_cells))
                # allocate grad_b for each layer
                grad_b1 = np.zeros((number_of_hidden_cells, 1))
                grad_b2 = np.zeros((BLOCK_DIMENSION*BLOCK_DIMENSION, 1))

                block_cost = sys.float_info.epsilon

                for block in batch:

                    block = np.reshape(block, (BLOCK_DIMENSION*BLOCK_DIMENSION, 1))

                    index = math.floor(i/128)

                    # print(index)

                    # compute the output (image is equal to block)
                    a1 = sigmoid((W1[index] @ block) + b1[index])
                    a2 = sigmoid((W2[index] @ a1) + b2[index])
            
                    # ---- Last layer
                    # weight
                    for j in range(grad_W2.shape[0]):
                        for k in range(grad_W2.shape[1]):
                            grad_W2[j, k] += 2 * (a2[j, 0] - block[j]) * a2[j, 0] * (1 - a2[j, 0]) * a1[k, 0]
                
                    # bias
                    for j in range(grad_b2.shape[0]):
                        grad_b2[j, 0] += 2 * (a2[j, 0] - block[j]) * a2[j, 0] * (1 - a2[j, 0])
                                    
                    # ---- 2nd layer
                    # activation
                    delta = np.zeros((number_of_hidden_cells, 1))
                    for k in range(number_of_hidden_cells):
                        for j in range(BLOCK_DIMENSION*BLOCK_DIMENSION):
                            delta[k, 0] += 2 * (a2[j, 0] - block[j]) * a2[j, 0] * (1 - a2[j, 0]) * W2[index][j, k]
            
                    # weight
                    for m in range(grad_W1.shape[0]):
                        for v in range(grad_W1.shape[1]):
                            grad_W1[m, v] += delta[m, 0] * a1[m,0] * (1 - a1[m, 0]) * block[v]
                
                    # bias
                    for m in range(grad_b1.shape[0]):
                        grad_b1[m, 0] += delta[m, 0] * a1[m, 0] * (1 - a1[m, 0])

            
                    W2 = W2 - (learning_rates[index] * (grad_W2))
                    W1 = W1 - (learning_rates[index] * (grad_W1))
                        
                    b2 = b2 - (learning_rates[index] * (grad_b2))
                    b1 = b1 - (learning_rates[index] * (grad_b1))

                    a1 = sigmoid(W1[index] @ block + b1[index])
                    a2 = sigmoid(W2[index] @ a1 + b2[index])

                    # 0 causes divided exception by zero in calculate_PSNR
                    # method so we initialize as a small number

                    for j in range(BLOCK_DIMENSION*BLOCK_DIMENSION):
                        block_cost += np.power((a2[j, 0] - block[j]), 2)

                    # print(block_cost)
                    i = (i+1)%1024

                    # print(i)
                    
        total_train_costs.append(block_cost)

    return W1, W2, b1, b2, total_train_costs


def momentum_train():

    momentum = 0.5
    psnr = 0
    minimum_psnr = 9.3
    epochs = 0

    previous_previous_W1 = []
    previous_previous_W2 = []
    previous_W1 = []
    previous_W2 = []

    previous_previous_b1 = []
    previous_previous_b2 = []
    previous_b1 = []
    previous_b2 = []

    W1 = []
    W2 = []
    b1 = []
    b2 = []
    learning_rates = []

    for i in range(number_of_networks):
        W1.append(np.random.normal(size=(number_of_hidden_cells, BLOCK_DIMENSION*BLOCK_DIMENSION)))
        W2.append(np.random.normal(size=(BLOCK_DIMENSION*BLOCK_DIMENSION, number_of_hidden_cells)))
        b1.append(np.random.normal(size=(number_of_hidden_cells, 1)))
        b2.append(np.random.normal(size=(BLOCK_DIMENSION*BLOCK_DIMENSION, 1)))
        learning_rates.append(random.uniform(0, 1))

    previous_previous_W1 = W1
    previous_previous_W2 = W2
    previous_previous_b1 = b1
    previous_previous_b2 = b2

    previous_W1 = W1
    previous_W2 = W2
    previous_b1 = b1
    previous_b2 = b2

    start = time.time()

    while(psnr < minimum_psnr):

        np.random.shuffle(train_set)
            
        for train_data in train_set:

            blocks = ressample(train_data, BLOCK_DIMENSION)

            batches = [blocks[x:(x+(math.floor(1024/number_of_networks)))] for x in range(0, len(blocks), (math.floor(1024/number_of_networks)))]

            i=0

            for batch in batches:

                grad_W1 = np.zeros((number_of_hidden_cells, BLOCK_DIMENSION*BLOCK_DIMENSION))
                grad_W2 = np.zeros((BLOCK_DIMENSION*BLOCK_DIMENSION, number_of_hidden_cells))
                # allocate grad_b for each layer
                grad_b1 = np.zeros((number_of_hidden_cells, 1))
                grad_b2 = np.zeros((BLOCK_DIMENSION*BLOCK_DIMENSION, 1))

                block_cost = sys.float_info.epsilon

                for block in batch:

                    block = np.reshape(block, (BLOCK_DIMENSION*BLOCK_DIMENSION, 1))

                    index = math.floor(i/128)

                    # print(index)

                    # compute the output (image is equal to block)
                    a1 = sigmoid((W1[index] @ block) + b1[index])
                    a2 = sigmoid((W2[index] @ a1) + b2[index])
            
                    # ---- Last layer
                    # weight
                    for j in range(grad_W2.shape[0]):
                        for k in range(grad_W2.shape[1]):
                            grad_W2[j, k] += 2 * (a2[j, 0] - block[j]) * a2[j, 0] * (1 - a2[j, 0]) * a1[k, 0]
                
                    # bias
                    for j in range(grad_b2.shape[0]):
                        grad_b2[j, 0] += 2 * (a2[j, 0] - block[j]) * a2[j, 0] * (1 - a2[j, 0])
                                    
                    # ---- 2nd layer
                    # activation
                    delta = np.zeros((number_of_hidden_cells, 1))
                    for k in range(number_of_hidden_cells):
                        for j in range(BLOCK_DIMENSION*BLOCK_DIMENSION):
                            delta[k, 0] += 2 * (a2[j, 0] - block[j]) * a2[j, 0] * (1 - a2[j, 0]) * W2[index][j, k]
            
                    # weight
                    for m in range(grad_W1.shape[0]):
                        for v in range(grad_W1.shape[1]):
                            grad_W1[m, v] += delta[m, 0] * a1[m,0] * (1 - a1[m, 0]) * block[v]
                
                    # bias
                    for m in range(grad_b1.shape[0]):
                        grad_b1[m, 0] += delta[m, 0] * a1[m, 0] * (1 - a1[m, 0])


                    W2 = W2 - (learning_rates[index] * (grad_W2) + momentum * (previous_W2 - previous_previous_W2))
                    W1 = W1 - (learning_rates[index] * (grad_W1) + momentum * (previous_W1 - previous_previous_W1))
                        
                    b2 = b2 - (learning_rates[index] * (grad_b2) + momentum * (previous_b2 - previous_previous_b2))
                    b1 = b1 - (learning_rates[index] * (grad_b1) + momentum * (previous_b1 - previous_previous_b1))

                    previous_previous_W2 = previous_W2
                    previous_previous_W1 = previous_W1
                    previous_previous_b2 = previous_b2
                    previous_previous_b1 = previous_b1

                    previous_W2 = W2
                    previous_W1 = W1
                    previous_b2 = b2
                    previous_b1 = b1

                    a1 = sigmoid(W1[index] @ block + b1[index])
                    a2 = sigmoid(W2[index] @ a1 + b2[index])

                    # 0 causes divided exception by zero in calculate_PSNR
                    # method so we initialize as a small number

                    for j in range(BLOCK_DIMENSION*BLOCK_DIMENSION):
                        block_cost += np.power((a2[j, 0] - block[j]), 2)

                    psnr = calculate_PSNR(block_cost)

                    # print(block_cost)
                    i = (i+1)%1024

                    # print(i)
                    
        total_train_costs.append(block_cost)
        epochs += 1    

    end = time.time()

    convergence_time = end - start

    print('Number of Iterations : ', epochs)

    print('Convergence Time is : ', convergence_time)

    return W1, W2, b1, b2, total_train_costs


def test(weight1, weight2, bias1, bias2):

    for test_data in test_set:

        output = np.zeros([256, 256]).astype(int)

        plt.imshow(test_data , cmap='gray')
        plt.show()

        i=0
        j=0
        counter = 0

        blocks = ressample(test_data, BLOCK_DIMENSION)

        for block in blocks:
            
            block = np.reshape(block, (BLOCK_DIMENSION*BLOCK_DIMENSION, 1))
            a1 = sigmoid((weight1 @ block) + bias1)
            a2 = sigmoid((weight2 @ a1) + bias2)


            # 0 causes divided exception by zero in calculate_PSNR
            # method so we initialize as a small number
            test_block_cost = sys.float_info.epsilon

            for j in range(BLOCK_DIMENSION*BLOCK_DIMENSION):
                test_block_cost += np.power((a2[j, 0] - block[j]), 2)

            test_block_psnr = calculate_PSNR(test_block_cost)

            total_test_psnrs.append(test_block_psnr)

            i_ratio = counter%32
            j_ratio = int(counter/32)
            for m in range(len(block)):
                for k in range(len(block[0])):
                    i_index = (BLOCK_DIMENSION*i_ratio) + i
                    j_index = (BLOCK_DIMENSION*j_ratio) + j
                    # print('i_ratio is : ', i_ratio)
                    # print('j_ratio is : ' , j_ratio)
                    # print('i_index is : ', i_index)
                    # print('j_index is : ' , j_index)
                    output[i_index][j_index] = block[m][k]
        
                    i = (i+1)%8
                j = (j+1)%8
            counter = counter+1

    output = np.transpose(output)
    
    plt.imshow(test_data , cmap='gray')
    plt.show()
    
    print('Avarage test PSNR is : ', (mean(total_test_psnrs)))



def plot_result(x, y, label_x, label_y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=label_x, ylabel=label_y, title='Histogram')
    ax.grid()
    fig.savefig("test.png")
    plt.show()


train_set = read_from_file('TrainSet')
test_set = read_from_file('TestSet')

W1, W2, b1, b2, total_costs = standard_train()
# W1, W2, b1, b2, total_costs = momentum_train()

epoch_size = [x for x in range(number_of_epochs)]
plot_result(epoch_size, total_costs, 'epoch size', 'Error')

test(W1, W2, b1, b2)