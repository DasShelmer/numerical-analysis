from copy import deepcopy

from numpy import abs, arctan, cos, sin, sqrt

SQRT2 = sqrt(2)


def init_matrix(size, fill=0):
    return [[fill for col in range(size)] for row in range(size)]


def print_matrix(matrix):
    for row in matrix:
        print(f"|{' '.join([str(a) for a in row])}|")


def max_element(matrix):
    max = -1
    maxI, maxJ = 0, 0
    for (row, i) in zip(matrix, range(len(matrix))):
        # Отступ i+1 означает выбор строго выше диагонали
        for (a, j) in zip(row[i+1:], range(i+1, len(row))):
            cur = abs(a)
            if i != j and cur > max:
                max = cur
                maxI = i
                maxJ = j

    return (maxI, maxJ)


def mul_Tmat(AT, B):
    N = len(AT)
    res = init_matrix(N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                res[i][j] += AT[k][i] * B[k][j]
    return res


def mul_mat(A, B):
    N = len(A)
    res = init_matrix(N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                res[i][j] += A[i][k] * B[k][j]
    return res


def calc_rot_matrix(matrix, maxI, maxJ):
    N = len(matrix)
    rot_mat = init_matrix(N)
    for i in range(N):
        rot_mat[i][i] = 1

    if matrix[maxI][maxI] == matrix[maxJ][maxJ]:
        rot_mat[maxI][maxI] = rot_mat[maxJ][maxJ] = rot_mat[maxJ][maxI] = SQRT2 / 2
        rot_mat[maxI][maxJ] = -SQRT2 / 2
    else:
        p = 2 * matrix[maxI][maxJ] / (matrix[maxI][maxI] - matrix[maxJ][maxJ])
        fi = 0.5 * arctan(p)
        rot_mat[maxI][maxI] = cos(fi)
        rot_mat[maxJ][maxJ] = cos(fi)
        rot_mat[maxI][maxJ] = -sin(fi)
        rot_mat[maxJ][maxI] = sin(fi)
    return rot_mat


def fault(matrix):
    sum = 0.0
    for (row, i) in zip(matrix, range(len(matrix))):
        # Выбор эл-ов строго выше диагонали
        for a in row[i+1:]:
            sum += a * a
    return sqrt(2 * sum)


def jacobi_run(matrix, eps):
    N = len(matrix)
    iteration = 0

    matrix = deepcopy(matrix)

    cur_fault = fault(matrix)

    rot_matrix = list()

    solution = init_matrix(N)
    for i in range(N):
        solution[i][i] = 1

    while cur_fault > eps:
        maxI, maxJ = max_element(matrix)
        rot_matrix = calc_rot_matrix(matrix, maxI, maxJ)

        temp = mul_Tmat(rot_matrix, matrix)
        matrix = mul_mat(temp, rot_matrix)
        cur_fault = fault(matrix)

        solution = mul_mat(solution, rot_matrix)

        iteration += 1

    eigenvalues = [matrix[i][i] for i in range(N)]
    eigenvectors = [[solution[j][i] for j in range(
        len(solution))] for i in range(len(solution[0]))]
    return (eigenvalues, eigenvectors, iteration, cur_fault)


def main():
    matrix = [
        [-2.612, 3.268, -4.505, -9.948, -2.137, 3.715, 0.498],
        [3.268, 9.208, -3.865, -8.143, -4.874, -2.64, -8.333],
        [-4.505, -3.865, -8.33, -6.356, -4.17, -2.985, -2.828],
        [-9.948, -8.143, -6.356, -7.267, 1.002, 2.383, 6.093],
        [-2.137, -4.874, -4.17, 1.002, -0.076, 7.526, 2.736],
        [3.715, -2.64, -2.985, 2.383, 7.526, 0.95, 1.629],
        [0.498, -8.333, -2.828, 6.093, 2.736, 1.629, -4.641]
    ]

    precision = 1e-4

    eigenvalues, eigenvectors, steps, fault = jacobi_run(matrix, precision)

    print('Решение:')
    for (eigenvector, i) in zip(eigenvectors, range(len(eigenvectors))):
        print(f"Собственный вектор k{i+1}")
        print(eigenvector)
    print('Собственные значения:')
    print(eigenvalues)
    print(f"Подсчитано за {steps} шагов")
    print(f"С точностью {fault}")


if __name__ == "__main__":
    main()
