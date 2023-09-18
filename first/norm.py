import numpy as np
from randomMatrix import randomMatrix, randomIntMatrix

if __name__ == "__main__":
    a = np.array([3, 4])
    #l1  норма
    l1 = np.linalg.norm(a, ord = 1)
    l2 = np.linalg.norm(a, ord = 2)
    l_inf = np.linalg.norm(a, ord = np.inf)
    l_p = np.linalg.norm(a, ord = 4)
    print(l1, l2, l_inf, l_p)

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    l_frob = np.linalg.norm(a, ord = "fro")
    l2 = np.linalg.norm(a, ord = 1)
    print(l_frob, l2)
    
