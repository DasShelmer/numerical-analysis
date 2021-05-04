

from numpy import cos, sin, sqrt


def print_matrix(A):
    """
    Вывод квадратной матрицы
    """
    full_mat = '\n'.join([f"|{' '.join([str(a) for a in row])}|" for row in A])
    print(full_mat)


def print_vector(V):
    """
    Вывод вектора
    """
    full_vec = f"X = ({' '.join([str(x) for x in V])})"
    print(full_vec)


def norma(A):
    """
    Вычисление нормы квадратной матрицы
    """
    n = 0

    for i in range(len(A)):
        for j in range(len(A[0])):
            n += A[i][j] * A[i][j]

    return sqrt(n)


def func(x):
    system = [
        sin(x[1] - 1.6) / 2,
        0.8 - cos(x[0]+0.5)
    ]
    return system.copy()


def jacobi(x):
    matrix = [
        [sin(x[0] + 0.5), 0],
        [0, 0.5 * cos(x[1])]
    ]
    return matrix.copy()


def main():
    N = 2
    eps = 1e-4
    a = list()
    x = list()
    x0 = [0, 0]
    iter = 0
    max = 9999999.

    while ((max > eps) and (iter < 20)):
        a = jacobi(x0)
        print_vector(x0)

        print(f"Norma = {norma(a)}")
        print(f"Iteration {iter}")
        print("=================")

        x = func(x0)
        max = abs(x[0] - x0[0])

        for i in range(N):
            if (abs(x[i] - x0[i]) > max):
                max = abs(x[i] - x0[i])

        x0 = x.copy()

        iter += 1


if __name__ == "__main__":
    main()
