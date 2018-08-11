
import numpy as np
import pickle
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W

# Replace this with your sigmoid implementation
def sigmoid(z):
    return (1 / (1 + (np.exp(-z))))
    
# Replace this with your nnObjFunction implementation
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
    #print(obj_val)

    # Regularization
    #--------------------    
    obj_val = obj_val + (lambdaval/(2*training_data.shape[0]))*(np.sum(np.sum(np.square(w1),axis=1),axis=0)+np.sum(np.sum(np.square(w2),axis=1),axis=0))
    #print(obj_val)

    # Gradient w2 
    #--------------------
    
    Difference_matrix = np.subtract(Ol, Yil)
    grad_w2 = (1.0/(training_data.shape[0]))*((np.dot(Difference_matrix.transpose(), Zj_w_bias)) + (lambdaval * w2)) 
    #print(grad_w2.shape)

    # Gradient w1
    #--------------------    
    Intermidiate_Exp = ((1. - Zj) * Zj) * (np.dot(Difference_matrix, w2[:,0:n_hidden]))
    grad_w1 = (1.0/(training_data.shape[0])) * ((np.dot(Intermidiate_Exp.transpose(), Train_data_w_bias)) + (lambdaval * w1))
    #print(grad_w1.shape)
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #print(obj_val)

    return (obj_val, obj_grad)

# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
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
# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
# Start timer to counr time for script
start = time.time()
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 75
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 15;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
end = time.time()
print("Execution time of script is " + str(end-start))

