import numpy as np

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))
    
# input sets
X = np.array(
           [[0,1,1,0,0,0,1,0,0,0,1,1,0,0,0],
[1,0,0,0,1,0,1,0,1,0,1,1,0,1,1],
[0,1,1,0,1,1,0,1,1,1,0,1,1,0,1],
[1,1,0,1,0,0,1,1,0,1,0,1,0,0,0],
[1,0,1,1,1,0,1,1,0,1,1,0,1,0,0],
[1,1,0,1,1,1,1,0,0,1,1,0,1,1,0]])

# output sets   
y = np.array(
           [[1,0,0,1,1,1,0,1,1,1,0,0,1,1,1],
[0,1,1,1,0,1,0,1,0,1,0,0,1,0,0],
[1,0,0,1,0,0,1,0,0,0,1,0,0,1,0],
[0,0,1,0,1,1,0,0,1,0,1,0,1,1,1],
[0,1,0,0,0,1,0,0,1,0,0,1,0,1,1],
[0,0,1,0,0,0,0,1,1,0,0,1,0,0,1]])

#tests
z = np.array(
           [[0,1,1,0,0,0,1,0,0,0,1,1,0,0,0]])

# seed for number randomizing
np.random.seed(1)

# random initialization of weights
weights0 = 2*np.random.random((15, 6)) - 1
weights1 = 2*np.random.random((6, 15)) - 1

for j in range(60000):

    # feed forward through the network
    level0 = X
    level1 = sigmoid(np.dot(level0,weights0))
    level2 = sigmoid(np.dot(level1,weights1))

    # calculating the error
    level2_error = y - level2
    
    # displaying the error every 10000 iterations
    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(level2_error))))
        
    # amount of change needed in the weights in the second layer
    level2_delta = level2_error*sigmoid(level2,deriv=True)

    # how much the first layer contributed in the error of the second layer
    level1_error = level2_delta.dot(weights1.T)
    
    # amount of change needed in the weights in the first layer
    level1_delta = level1_error * sigmoid(level1,deriv=True)

    # change of weights
    weights1 += level1.T.dot(level2_delta)
    weights0 += level0.T.dot(level1_delta)

#testing examples that did not appear in the training process
level0 = z
level1 = sigmoid(np.dot(level0,weights0))
level2 = sigmoid(np.dot(level1,weights1))

print(np.rint(level2))