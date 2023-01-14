import numpy as np


# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0231053X(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    # your code goes here
    a = 1.5
    b = 0.3
    c = 1
    d = 2
    a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out = [], [], [], [], [], [], []
    
    for i in range(num_iters):
        grad_a = 4*a**3
        grad_b = 2*np.sin(b)*np.cos(b)
        grad_c = 2*c
        grad_d = d*(2*np.sin(d) + d*np.cos(d))
        
        a -= learning_rate*grad_a
        a_out.append(a)
        f1_out.append(a**4)
        
        b -= learning_rate*grad_b
        b_out.append(b)
        f2_out.append((np.sin(b))**2)
        
        c -= learning_rate*grad_c
        d -= learning_rate*grad_d
        c_out.append(c)
        d_out.append(d)
        f3_out.append(c**2 + d**2 * np.sin(d))

    a_out = np.array(a_out)
    f1_out = np.array(f1_out)
    b_out = np.array(b_out)
    f2_out = np.array(f2_out)
    c_out = np.array(c_out)
    d_out = np.array(d_out)
    f3_out = np.array(f3_out)
    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out