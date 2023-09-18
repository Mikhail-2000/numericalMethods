import numpy as np
from randomMatrix import randomMatrix

def myKrammerSolution(a: np.ndarray, b: np.ndarray)-> np.ndarray:
    det = np.linalg.det(a)

    if abs(det) < 1e-9:
        raise RuntimeError("No Solution")
    else:
        x = np.zeros((n, 1))
        #print(x)
        for i in range(0, n):
            a_i = np.copy(a)
            #print(a_i[:, i].shape)
            a_i[:, i] = b[:, 0]
            x[i, 0] = np.linalg.det(a_i) / det
        return x
def numpySolution(a: np.ndarray, b: np.ndarray)-> np.ndarray:
    return np.linalg.solve(a, b)


if __name__ == "__main__":
    n = 5
    a = randomMatrix(n, n)
    b = randomMatrix(n, 1)
    x1 = myKrammerSolution(a, b)
    x2 = numpySolution(a, b)
    #print(x1)
    #print(x2)
    print(x1 - x2)


