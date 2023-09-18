import numpy as np
def randomMatrix(x: int, y: int) -> np.ndarray:
    return np.full((x, y), 0.5) - np.random.rand(x, y)

def randomIntMatrix(x: int, y: int, low: int = -9, hight: int = 9) -> np.ndarray:
    return np.random.randint(low, hight, (x, y))

if __name__ == "__main__":
    print(randomMatrix(3, 3))
    print(randomIntMatrix(4, 4))