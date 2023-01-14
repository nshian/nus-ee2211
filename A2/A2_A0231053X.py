import numpy as np
from numpy.linalg import inv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0231053X(N):
    """
    Input type
    :N type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """
    # your code goes here
    data = load_iris()
    X = data['data']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=N)
    Ytr = y_train.reshape(-1,1)
    Yts = y_test.reshape(-1,1)
    
    enc = OneHotEncoder(sparse=False)
    Ytr = enc.fit_transform(Ytr)
    Yts = enc.fit_transform(Yts)
    
    Ptrain_list = []
    Ptest_list = []
    w_list = []
    error_train_array = []
    error_test_array = []
    for i in range(1, 11):
        poly = PolynomialFeatures(i)
        Ptrain = poly.fit_transform(X_train)
        Ptrain_list.append(Ptrain)
        Ptest = poly.fit_transform(X_test)
        Ptest_list.append(Ptest)
        
        m, d = Ptrain.shape[0], Ptrain.shape[1]
        factor = 0.0001
        if m > d: #Primal form
            PTP = Ptrain.T @ Ptrain
            reg = factor*np.identity(PTP.shape[0])
            w = inv(PTP + reg) @ Ptrain.T @ Ytr
        else:
            PPT = Ptrain @ Ptrain.T
            reg = factor*np.identity(PPT.shape[0])
            w = Ptrain.T @ inv(PPT + reg) @ Ytr
        
        w_list.append(w)
        pred_train = Ptrain @ w
        pred_test = Ptest @ w
        error_train_array.append(sum([0 if pred_train[i].argmax() == Ytr[i].argmax() else 1 for i in range(len(Ytr))]))
        error_test_array.append(sum([0 if pred_test[i].argmax() == Yts[i].argmax() else 1 for i in range(len(Yts))]))
    
    error_train_array = np.array(error_train_array)
    error_test_array = np.array(error_test_array)

    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array
