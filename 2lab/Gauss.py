import numpy as np
import math
import matplotlib.pyplot as plt

#Аккуратный вывод матрицы
def AccuratePrint(a: np.ndarray):
    with np.printoptions(precision=3):
        print(a)
    print()

#Генерирует случайную матрицу с числами от -0.5 до 0.5
def getRandomMatrix(n: int) -> np.ndarray:
    return np.full((n, n), 0.5) - np.random.rand(n, n)

#Генерирует случайную систему A с единичным решением
def GetRandomSystem(n: int) -> (np.ndarray, np.ndarray, np.ndarray):
    A = getRandomMatrix(n)
    b = np.zeros((n, 1))
    for i in range(n):
        b[i] += np.sum(A[i, :])
    x = np.zeros((n, 1))
    return (A, x, b)

#Решение методом Гаусса без выбора ведущего элемента
def gaussSolveWithoutPivoting(A:np.ndarray, b: np.ndarray):
    #swaps = 0
    n, m = A.shape
    if n != m:
        raise RuntimeError("n and m should be equals")
    A_b = np.concatenate((A, b), axis = 1)
    A = A_b

    #straight running
    for k in range(n):
        akk = A[k, k]
        tmp = A[k, :] / akk
        for i in range(k+1, n):
            A[i, :] = A[i, :] - tmp * A[i, k]
    print("Приведенная расширенная матрица без перестановок")
    AccuratePrint(A)
    #backward running
    x = np.zeros((n, 1))
    for k in range(n-1, -1, -1):
        ans = A[k, n]
        #print(ans)
        ans -= np.sum(A[k,:n] * x.T)
        ans /= A[k,k]
        x[k, 0] = ans
    return x

#Решение методом Гаусса с выбором ведущего элемента
def gaussSolveWithPivoting(A:np.ndarray, b: np.ndarray):
    swaps = 0
    n, m = A.shape
    if n != m:
        raise RuntimeError("n and m should be equals")
    A_b = np.concatenate((A, b), axis = 1)
    A = A_b
    #straight running
    for k in range(n):
        opt_pos = k
        for i in range(k, n):
            if abs(A[opt_pos, k]) < abs(A[i, k]):
                opt_pos = i
        if opt_pos != k:
            A[[k, opt_pos], :] = A[[opt_pos, k], :]
            swaps += 1
        akk = A[k, k]
        tmp = A[k, :] / akk
        for i in range(k+1, n):
            A[i, :] = A[i, :] - tmp * A[i, k]
    print("Приведенная расширенная матрица с перестановками")
    AccuratePrint(A)
    #backward running
    x = np.zeros((n, 1))
    for k in range(n-1, -1, -1):
        ans = A[k, n]
        ans -= np.sum(A[k,:n] * x.T)
        ans /= A[k,k]
        x[k, 0] = ans
    return (x, swaps)


n = 1000                     #Размер матрицы
A, x, b = GetRandomSystem(n)
#AccuratePrint(A)
#AccuratePrint(np.concatenate((A, b), axis = 1))
print("Det(A): {:.3f}".format(np.linalg.det(A)) )
solve0 = gaussSolveWithoutPivoting(A, b)
#print("Решение системы методом Гаусса:")
#print(solve0)

(solve1, swaps) = gaussSolveWithPivoting(A, b)
#print("Решение системы методом Гаусса с перестановкой строк:")
#print(solve1)
print("Число перестановок строк: %s" % (swaps))
ones = np.ones((n, 1))

print("Разность решений без выбора ведущего элемента (2 норма): %s" % np.linalg.norm(solve0-ones))
print("Разность решений с выбором ведущего элемента (2 норма): %s" % np.linalg.norm(solve1-ones))