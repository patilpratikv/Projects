import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def preprocess():
    """ 
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    # Creating bias term
    bias_term = np.ones((n_data,1))
    # Concatenating bias term to the input feature vector
    train_data_w_bias = np.concatenate((bias_term, train_data),axis=1)
    # Initializing theta vector
    theta = np.zeros((train_data_w_bias.shape[0],1))
    # Reshaping weight vector to column vector
    Weights_reshaped = initialWeights.reshape(train_data_w_bias.shape[1],1)
    # Calculating Wtranspose * X
    wt_x = np.dot(train_data_w_bias, Weights_reshaped)
    # Passing the multiplication value to activation sigmoid function
    theta = sigmoid(wt_x)  
    # Calculating error value with the formula given in the pdf
    error=(-np.sum(np.dot(labeli.transpose(),np.log(theta)) + np.dot(np.subtract(1.0,labeli).transpose(),np.log(np.subtract(1.0,theta)))))/train_data_w_bias.shape[0] 
    # Calcuating error gradient value with the formula shared in the pdf   
    error_grad = (np.dot(train_data_w_bias.transpose(),np.subtract(theta,labeli))/train_data_w_bias.shape[0]).flatten()

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    # Creating bias term
    bias_term = np.ones((data.shape[0],1))
    # Concatenating bias term to the input feature vector
    train_data_w_bias = np.concatenate((bias_term,data),axis=1)
    # Calculating Wtranspose * X
    wt_x = np.dot(train_data_w_bias, W)
    # Passing the multiplication value to activation sigmoid function
    label_prediction = sigmoid(wt_x)
    # Find the index of largest value in the array that is created
    label = np.argmax(label_prediction, axis = 1)
    # Reshaping labels into column vector
    label = label.reshape(data.shape[0],1)

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    # params in a (7160,) shape, reshaping it to (716,10) matrix
    Weights = params[:].reshape((n_feature+1,n_class))
    # Creating bias term
    Bias_Term = np.ones((train_data.shape[0],1))
    # Concatenating bias term to the input feature vector
    train_data = np.concatenate((Bias_Term, train_data), axis = 1)
    # Calculating exp(W*X)
    e_Wt_x = np.exp(np.dot(train_data,Weights))
    # Calculating softmax
    soft_max = e_Wt_x/np.sum(e_Wt_x,axis=1).reshape(train_data.shape[0],1)
    # Calculating error with respect to actual labels
    error = np.sum(-(labeli*np.log(soft_max)))
    # Calculating error gradient
    error_grad = (np.dot(train_data.transpose(), soft_max-labeli)/train_data.shape[0]).flatten()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    # Creating bias term
    bias_term = np.ones((data.shape[0],1))  
    # Concatenating bias term to the input feature vector
    train_data_w_bias = np.concatenate((bias_term, data),axis=1) 
    # find the activation function value after multiplying with weights which are already learnt in training section.
    prediction = sigmoid(np.dot(train_data_w_bias, W)) 
    # Find the maximum value in an array after activation
    label = np.argmax(prediction, axis = 1).reshape(data.shape[0],1)

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label_training = predicted_label

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label_validation = predicted_label

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
predicted_label_test = predicted_label

dataset_strings = ['Training Data', 'Validation Data', 'Test data']
true_labels = [train_label, validation_label, test_label]
predicted_label = [predicted_label_training, predicted_label_validation, predicted_label_test]

print('Confusion matrix for Logistic Regression :\n')
for i in range(len(true_labels)):
    print('Confusion matrix for ' + dataset_strings[i] + '\n')
    
    cm = confusion_matrix(y_true=true_labels[i], y_pred=predicted_label[i])

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(n_class)
    plt.xticks(tick_marks, range(n_class))
    plt.yticks(tick_marks, range(n_class))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

"""
Script for Support Vector Machine
"""
print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

#linear kernel
print("Results for Linear Kernel")
svc_kernel = SVC(kernel = 'linear')
svc_kernel.fit(train_data, train_label)
print("Training set Accuracy: ",svc_kernel.score(train_data, train_label))
print("Validation set Accuracy: ",svc_kernel.score(validation_data, validation_label))
print("Test set Accuracy: ",svc_kernel.score(test_data, test_label))

#-----------------------------------------------------------------------------

#kernel with gamma = 1
print("Results for kernel with gamma = 1")
svc_kernel = SVC(gamma = 1)
svc_kernel.fit(train_data, train_label)
print("Training set Accuracy: ",svc_kernel.score(train_data, train_label))
print("Validation set Accuracy: ",svc_kernel.score(validation_data, validation_label))
print("Test set Accuracy: ",svc_kernel.score(test_data, test_label))

#-----------------------------------------------------------------------------

#kernel with default parameters
print("Results for kernel with default parameters")
svc_kernel = SVC(kernel = 'rbf')
svc_kernel.fit(train_data, train_label)
print("Training set Accuracy: ",svc_kernel.score(train_data, train_label))
print("Validation set Accuracy: ",svc_kernel.score(validation_data, validation_label))
print("Test set Accuracy: ",svc_kernel.score(test_data, test_label))

#-----------------------------------------------------------------------------

# Find the accuracy on Training Dataset# Creating array for the values of C
C_array = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
C_array_train_data = []
C_array_validate_data = []
C_array_test_data = []
# Loop through all values of C
for i in C_array:
    print("Calculating for C = " + str(i))
    # Creating instance of SVC
    svc_kernel = SVC(C = i)
    # Training SVC model
    svc_kernel.fit(train_data, train_label)
    # Calculating accuracy for training data
    accuracy_of_train_data = svc_kernel.score(train_data, train_label)
    print("Training Set Accuracy: ",accuracy_of_train_data)
    # Calculating accuracy for validation data
    accuracy_of_validation_data = svc_kernel.score(validation_data, validation_label)
    print("Validation set Accuracy: ",accuracy_of_validation_data)
    # Calculating accuracy for test data
    accuracy_of_test_data = svc_kernel.score(test_data, test_label)
    print("Test set Accuracy: ",accuracy_of_test_data)
    # Converting accuracy in percentages for all
    C_array_train_data.append(100 * accuracy_of_train_data)
    C_array_validate_data.append(100 * accuracy_of_validation_data)
    C_array_test_data.append(100 * accuracy_of_test_data)

Acc_in_column_vector = np.column_stack((C_array_train_data, C_array_validate_data, C_array_test_data))
# Plotting graph of accuracies for each dataset for different values of C 
plt.figure(figsize=[10,6])
plt.subplot(1, 2, 1)
plt.plot(C_array, Acc_in_column_vector)
plt.title('C Values Vs. Accuracy')
plt.legend(('Training data', 'Validation data', 'Testing data'), loc = 'best') 
plt.xlabel('C')
plt.ylabel('Accuracy in percentage') 
plt.show()

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
predicted_label_train = predicted_label_b

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')
predicted_label_validation = predicted_label_b

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
predicted_label_test = predicted_label_b

predicted_label = [predicted_label_train, predicted_label_validation, predicted_label_test]

print('Confusion matrix for Multi Class Logistic Regression :\n')

for i in range(len(true_labels)):
    print('Confusion matrix for ' + dataset_strings[i] + '\n')
    
    cm = confusion_matrix(y_true=true_labels[i], y_pred=predicted_label[i])

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(n_class)
    plt.xticks(tick_marks, range(n_class))
    plt.yticks(tick_marks, range(n_class))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()