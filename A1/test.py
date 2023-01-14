import A1_A0231053X as grading
import numpy as np

X = np.array([[1,4], [4,2], [5,6], [3,-3], [9,-10]])
y= np.array([-1,2,1,0,4])
InvXTX, w = grading.A1_A0231053X(X,y)

print(InvXTX)
print(w)