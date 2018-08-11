
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
import time

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  (1 / (1 + (np.exp(-z))))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    global feature_column_inidices
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    # Comparing first row of matrix with all other rows of matrix
    compare_train_matrix = train_data == train_data[0,:]
    # Compare along different columns
    indentical_column_bool_array = np.all(compare_train_matrix, axis = 0)
    # Find the index of column where they are identicle
    identical_column_indices = np.where(indentical_column_bool_array)[0]
    # Find indices of colums which can be used as features
    feature_column_inidices = np.where(np.logical_not(indentical_column_bool_array))[0]
    # delete all coumns in training data, validation data and test data which are identicle across all images 
    # and useless for prediction
    train_data = np.delete(train_data, identical_column_indices, 1)
    validation_data = np.delete(validation_data, identical_column_indices, 1)
    test_data = np.delete(test_data, identical_column_indices, 1)
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    # Constructing bias term
    Input_Bias_Term = np.ones((training_data.shape[0],1))
    # Concatenate bias term to the train dataset along column
    Train_data_w_bias = np.concatenate((training_data, Input_Bias_Term), axis = 1)
    # Multplying train data features with weight matrix    
    Aj = np.dot(Train_data_w_bias, w1.transpose())
    # Calculating sigmoid value for each unit of hidden layer
    Zj = sigmoid(Aj)     
    # Constructing bias term for hidden layer
    Hidden_Layer_Bias = np.ones((Zj.shape[0],1))
    # Cancatenating it with previous calculated sigmoid function
    Zj_w_bias = np.concatenate((Zj, Hidden_Layer_Bias),axis=1)
    # Multiplying it with weight matrix of output layer and calculating sigmoid value for the same
    Bl = np.dot(Zj_w_bias, w2.transpose())
    Ol = sigmoid(Bl)            
    # Constructing Yil matrix with the help of training labeled data.
    # For each image construction a row of 10 elements where all elemts will be zero excepts its labeled number index
    Yil = np.zeros(Ol.shape)
    for i in range(Yil.shape[0]): 
        Yil[i][int(training_label[i])] = 1.0        
     

    # Error function
    #--------------------    
    obj_val = (-1.0/training_data.shape[0])*(np.sum((Yil*np.log(Ol))+(1.-Yil)*np.log(1. - Ol)))
    
    # Regularization
    #--------------------    
    obj_val = obj_val + (lambdaval/(2*training_data.shape[0]))*(np.sum(np.sum(np.square(w1),axis=1),axis=0)+np.sum(np.sum(np.square(w2),axis=1),axis=0))
    
    # Gradient w2 
    #--------------------
    
    Difference_matrix = np.subtract(Ol, Yil)
    grad_w2 = (1.0/(training_data.shape[0]))*((np.dot(Difference_matrix.transpose(), Zj_w_bias)) + (lambdaval * w2)) 
    
    # Gradient w1
    #--------------------    
    Intermidiate_Exp = ((1. - Zj) * Zj) * (np.dot(Difference_matrix, w2[:,0:n_hidden]))
    grad_w1 = (1.0/(training_data.shape[0])) * ((np.dot(Intermidiate_Exp.transpose(), Train_data_w_bias)) + (lambdaval * w1))
        
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    
    # Constructing bias term
    Input_Bias_Term = np.ones((data.shape[0],1))
    # Concatenate bias term to the input data along column
    Input_w_bias = np.concatenate((data, Input_Bias_Term), axis = 1)    
    # Multiply data with trained weights of hidden layer
    Aj = np.dot(Input_w_bias, w1.transpose())
    # Calculate sigmoid for multiplied value
    Zj = sigmoid(Aj)     
    # Constructing bias term
    Hidden_Layer_Bias_Term = np.ones((Zj.shape[0],1))
    # Concatenate bias term to the input data along column
    Zj_w_bias = np.concatenate((Zj, Hidden_Layer_Bias_Term),axis=1)
    # Multiply data with trained weights of output layer
    Bl = np.dot(Zj_w_bias,w2.transpose())
    # Calculate sigmoid for multiplied value
    Ol = sigmoid(Bl)
    # argmax can find the maximum value index along row, which is actually the number predicted for image
    labels = np.argmax(Ol, axis = 1)
    
    return labels


"""**************Neural Network Script Starts here********************************"""


# Start timer to counr time for script
start = time.time()    

feature_column_inidices = [];

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 75

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 15

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

all_data = feature_column_inidices, n_hidden, w1, w2, lambdaval

pickle.dump(all_data, open('params.pickle', 'wb'))

end = time.time()
print("Execution time of script is " + str(end-start))

