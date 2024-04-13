import numpy as np
import math
import matplotlib.pyplot as plt

def accuratePrint(A: np.ndarray):
    with np.printoptions(precision=3):
        print(A)
    print()
#генерирует случайную матрицу с числами от -0.5 до 0.5
def getRandomMatrix(n: int) -> np.ndarray:
    return np.full((n, n), 0.5) - np.random.rand(n, n)

#Генерирует случайную положительно определенную матрицу A с единичным решением
def GetRandomSystem(n: int) -> (np.ndarray, np.ndarray, np.ndarray):
    A = getRandomMatrix(n)
    A = np.dot(A.T, A) 
    b = np.zeros((n, 1))
    for i in range(n):
        b[i] += np.sum(A[i, :])
    x = np.zeros((n, 1))
    return (A, x, b)

#решение СЛАУ методом верхней релаксации
#(D + Lt)@(x_(k+1) - x_k)/t + (L + D + U)*x_k = b
def SOR(A: np.ndarray, 
        b: np.ndarray, 
        x0: np.ndarray, 
        eps: float, 
        max_iter: int,
        t: int) -> (np.ndarray, int):
    
    if (not (1 <= t and t <= 2)): #проверка фактора
        raise RuntimeError('t should be inside [1, 2]')
    
    n = b.shape[0] 
    #b(n, 1)
    x = x0 #временный массив
    last_step = 0
    for step in range (1, max_iter): #итерации
        last_step = step
        for i in range(n):
            tmp1 = np.dot(A[i, 0:i], x[:i])
            tmp2 = np.dot(A[i, i+1 :], x0[i+1:])
            x[i] = (1-t) * x0[i] + t/A[i, i] * (b[i] - tmp1 - tmp2)
        if np.linalg.norm(np.dot(A, x)-b ) < eps:
            #print(*x)
            break
    return (x, last_step)

#Строим графики зависимости числа итераций от значения параметра
#Для системы Ax=b, с шагом step
def getPlot(A: np.ndarray, b: np.ndarray, step: int = 0.05):
    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump
    global eps, max_iter
    x0 = np.zeros((n, 1))
    iter_cnt = []
    t_value = []
    for t in frange(1.0, 2.0, step):
        iter_cnt.append(SOR(A, b, x0, eps, max_iter, t)[1])
        t_value.append(t)
        x0 = np.zeros((n, 1))
    accuratePrint(iter_cnt)
    print(np.min(iter_cnt))
    plt.plot(t_value, iter_cnt)
    plt.title("графики зависимости числа итераций от значения параметра")
    plt.xlabel("t: value of parameter")
    plt.ylabel("Number of iteration")
    plt.show()

n = 5                   #рaзмер матрицы
eps = 1e-9              #точность
max_iter = 5000         #максимальное число итераций

A, x0, b = GetRandomSystem(n)
accuratePrint(A)
Ab = np.concatenate((A, b), axis = 1)
accuratePrint(Ab)
x1 = SOR(A, b, x0, eps, max_iter, t = 1)[0]
print(x1)
getPlot(A, b, step = 0.01)

