import numpy as np
from randomMatrix import randomMatrix, randomIntMatrix

def myDot(a:np.ndarray, b:np.ndarray)-> np.ndarray:
    if a.shape[1] != b.shape[0]:
        raise RuntimeError("Incorrect Size")
    c = np.zeros( (a.shape[0], b.shape[1]) )
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for pos in range(a.shape[1]):
                c[i, j] += a[i, pos] * b[pos, j]
    return c

def numpyDot(a:np.ndarray, b:np.ndarray)-> np.ndarray:
    #return a.dot(b)
    return np.dot(a, b)

if __name__ == "__main__":
    a = randomIntMatrix(3, 5)
    b = randomIntMatrix(3, 5)
    b = b.T
    c_np = numpyDot(a, b)
    print(c_np)

    c = myDot(a, b)
    print(c)
