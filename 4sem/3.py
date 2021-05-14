from random import randint

from numpy import abs, arctan, cos, dot, mat, sin, sqrt

SQRT2 = sqrt(2)

# Вспомогательные функции


def r():
    """Генерирует случайное число [-10.000; 10.000]"""
    return randint(-10000, 10000) / 1000


def make_symmetrical(A, by_upper=True):
    """
    Делает матрицу симметричной на снове
    копирования эл-тов выше/ниже диагонали
    в соответствующие эл-ты ниже/выше диагонали
    """
    for i in range(len(A)):
        if by_upper:
            for j in range(0, i):
                A[i][j] = A[j][i]
        else:
            for j in range(i+1, len(A)):
                A[j][i] = A[i][j]


def is_symmetrical(A):
    """
    Проверяет является ли матрица симметричной
    относительно главной диагонали
    """
    for i in range(len(A)):
        for j in range(0, i):
            if A[i][j] != A[j][i]:
                return False
    return True


def random_matrix(size=7, symmetrical=True):
    """Генерирует матрицу со случайными эл-ми"""
    mat = [[r() for _ in range(size)] for _ in range(size)]
    if symmetrical:
        make_symmetrical(mat)
    return mat


def print_matrix(matrix, adv_text=None):
    """Вывод матрицы"""
    if adv_text:
        print(adv_text)

    for row in matrix:
        print(f"|{' '.join([str(a) for a in row])}|")
# Конец вспомогательных функций


def init_matrix(size, fill=0):
    """Создание матрицы с заполнением всех эл-ов"""
    return [[fill for _ in range(size)] for _ in range(size)]


def max_element(matrix):
    """Нахождение индексов максимального эл-та матрицы"""
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


def t_mul_mat(AT, B):
    """Перемножение матриц, где первая транспонируется (A^T x B)"""
    N = len(AT)
    res = init_matrix(N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # Транспонирование заключается в смене пары индексов AT[i][k] на AT[k][i]
                res[i][j] += AT[k][i] * B[k][j]
    return res


def mul_mat(A, B):
    """Перемножение матриц (A x B)"""
    res = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                res[i][j] += A[i][k] * B[k][j]
    return res


def calc_rot_matrix(matrix, maxI, maxJ):
    """Построение матрицы поворота"""
    N = len(matrix)
    rot_mat = init_matrix(N)
    # Создание единичной матрицы
    for i in range(N):
        rot_mat[i][i] = 1

    # Частный случай, где fi = pi/4
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
    """Вычисление погрешности"""
    sum = 0.0
    for (row, i) in zip(matrix, range(len(matrix))):
        # Выбор эл-ов строго выше диагонали
        for a in row[i+1:]:
            sum += a * a
    return sqrt(2 * sum)


def jacobi(matrix, precision):
    """
    Метод вращения Якоби для нахождения
    собственных значений матрицы и
    собственных векторов с заданной точностью
    """

    N = len(matrix)

    # Кол-во итераций (вращений)
    iteration = 0

    # Полное копирование исходной матрицы
    matrix = [[a for a in row] for row in matrix]

    # Расчёт текущей погрешности
    cur_fault = fault(matrix)

    # Матрица с помощью кот. осуществляем поворот
    rot_matrix = list()

    # Матрица, где столбцы - это собственные векторы
    solution = init_matrix(N)
    for i in range(N):
        solution[i][i] = 1

    # Выполняем пока погрешность не будет приемлимой
    while cur_fault > precision:
        maxI, maxJ = max_element(matrix)
        rot_matrix = calc_rot_matrix(matrix, maxI, maxJ)

        # Первая фаза поворота
        temp = t_mul_mat(rot_matrix, matrix)
        # Вторая фаза поворота
        matrix = mul_mat(temp, rot_matrix)

        # Обновляем собственные векторы на основе матрицы поворота
        solution = mul_mat(solution, rot_matrix)

        # Расчёт текущей погрешности
        cur_fault = fault(matrix)
        iteration += 1

    # Собираем с.з. из диагонали матрицы в вектор
    eigenvalues = [matrix[i][i] for i in range(N)]
    # Транспонируем с.в. (т.к. они записаны столбцами их неудобно обрабатывать дальше)
    eigenvectors = [[solution[j][i] for j in range(
        len(solution))] for i in range(len(solution[0]))]

    return (eigenvalues, eigenvectors, iteration, cur_fault)


def mul_mat_vect(A, V):
    res = [0]*len(A[0])

    for i in range(len(A[0])):
        for j in range(len(V)):
            res[i] += V[j] * A[j][i]

    return res


def vect_dif(A, B):
    return [x-y for (x, y) in zip(A, B)]


def is_answer(A, eigenvalues, eigenvectors):
    # A x v - λ x v
    for (vector, value) in zip(eigenvectors, eigenvalues):
        AV = mul_mat_vect(A, vector)
        lambdaV = [value * x for x in vector]
        diff = vect_dif(AV, lambdaV)
        print(diff)


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

    eigenvalues, eigenvectors, steps, fault = jacobi(matrix, precision)

    print('Решение:')

    for (eigenvector, i) in zip(eigenvectors, range(len(eigenvectors))):
        print(f"Собственный вектор k{i+1}")
        print(eigenvector)

    print('Собственные значения:')
    print(eigenvalues)

    print(f"Подсчитано за {steps} шагов")
    print(f"С точностью {fault}")
    is_answer(matrix, eigenvalues, eigenvectors)


if __name__ == "__main__":
    main()
