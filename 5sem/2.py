from math import *

import numpy as np

# решение
y_x = lambda x: -1 + (2 + 2 * (x + 1) * np.log(abs(x + 1))) / x


q_x = lambda x: -2 / (x * x * (x + 1))
p_x = lambda x: 0
f_x = lambda x: (2 - 2 * x) / (x * x * (x + 1))


# Метод прогонки
def tridig_matrix_alg(A, b):
    P = [-item[2] for item in A]
    Q = [item for item in b]
    P[0] /= A[0][1]
    Q[0] /= A[0][1]
    for i in range(1, len(b)):
        z = A[i][1] + A[i][0] * P[i - 1]
        P[i] /= z
        Q[i] -= A[i][0] * Q[i - 1]
        Q[i] /= z

    x = [item for item in Q]
    for i in range(len(x) - 2, -1, -1):
        x[i] += P[i] * x[i + 1]
    return x


def find_tridig_A(h, p, q, x):
    # Строим трёхдиагональную матрицу как множество триад
    # [[0, a_11, 0], [a_21, a_22, a_12], ..., [a_n(n-1), a_nn, a_(n-1)n]]
    A = [
        [1 - p(x[i]) / 2, (-2 + h * h * q(x[i])), 1 + (p(x[i]) * h) / 2]
        for i in range(1, len(x[:-1]))
    ]
    A[0][0] = 0
    A[-1][-1] = 0
    return A


def find_b(h, p, f, x, y0, y1):
    b = [h * h * f(x[i]) for i in range(1, len(x[:-1]))]
    b[0] -= y0 * (1 - p(x[1]) * h / 2)
    b[-1] -= y1 * (1 + p(x[-2]) * h / 2)
    return b


def FD(a, b, n, y0, y1, p=p_x, q=q_x, f=f_x):
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]  # [a, a+h, a+2h, ..., b-h, b]
    A = find_tridig_A(h, p, q, x)
    b = find_b(h, p, f, x, y0, y1)
    T = [y0] + tridig_matrix_alg(A, b) + [y1]
    return T


def max_diff(T: list, T_: list):
    # Находит максимальную разницу между
    # каждым значением из T и
    # каждым вторым значением из T_
    T_second = T_[::2]  # Каждый второй элемент с индекса 0
    diff = [abs(a - b) for (a, b) in zip(T, T_second)]
    index = np.argmax(diff)
    return (index, diff[index])


def valf(v):
    # Преобразует число в текст,
    # если вместо числа None,
    # то возвращает "---"
    float_value = "{:20.10}"
    none_value = "".join([" "] * 5) + "".join(["-"] * 15)
    return float_value.format(v) if not v is None else none_value


def fshow(T, T_, n, a, b):
    # Красивый вывод таблички

    offset = 5
    indexes = [
        *range(offset),  # Начало
        -1,
        *range(n // 2 - offset // 2, n // 2 + offset // 2),  # Середина
        -1,
        *range(n - (offset - 1), n),  # Конец
    ]
    # indexes = range(n + 1) # Всё

    print("         i", end=" ")
    print("                x[i]", end=" ")
    print("                 y_h", end=" ")
    print("             y_(h/2)", end=" ")
    print("              y_real", end=" ")
    print("     |y_h - y_(h/2)|", end=" ")
    print("  |y_(h/2) - y_real|")

    for i in indexes:
        if i < 0:
            print("".join(["."] * 136))
            continue

        x_i = a + i * (b - a) / (n - 1)
        y_h = T[i // 2] if i % 2 == 0 else None
        y_h2 = T_[i]
        y_real = y_x(x_i)
        diff_y_h_y_h2 = abs(y_h - y_h2) if i % 2 == 0 else None
        diff_y_h2_y_real = abs(y_h2 - y_real)

        print(f"{i:10}", end=" ")
        print(valf(x_i), end=" ")
        print(valf(y_h), end=" ")
        print(valf(y_h2), end=" ")
        print(valf(y_real), end=" ")
        print(valf(diff_y_h_y_h2), end=" ")
        print(valf(diff_y_h2_y_real))


def main():
    # переделать

    eps = 1e-4

    a = 1.0
    b = 2.0
    N = 4

    y0 = y_x(a)
    y1 = y_x(b)

    n = N
    subdiv = 0
    T = FD(a, b, n, y0, y1)
    n = 2 * n
    subdiv += 1
    T_ = FD(a, b, n, y0, y1)

    while max_diff(T, T_)[1] > eps:
        T = T_
        n = 2 * n
        subdiv += 1
        T_ = FD(a, b, n, y0, y1)

    fshow(T, T_, n + 1, a, b)

    T_real = [y_x(a + i * (b - a) / n) for i in range(n)]

    print("Количество уменьшений шага = ", subdiv)

    max_i, max_diff_val = max_diff(T, T_)
    max_x = a + max_i * (b - a) / (n - 1)
    max_y_h = T[max_i]
    max_y_h2 = T_[max_i * 2]
    print(
        "max|y_h - y_h2| =   ",
        max_diff_val,
    )
    print(
        "   i = ",
        max_i,
        "  x[i] = ",
        max_x,
        "  y_h[i] = ",
        max_y_h,
        "  y_h2 = ",
        max_y_h2,
    )

    rmax_i, rmax_diff_val = max_diff(T, T_real)
    rmax_x = a + rmax_i * (b - a) / (n - 1)
    rmax_y_h = T[rmax_i]
    max_y_real = T_real[rmax_i * 2]
    print("max|y_h - y_real| = ", rmax_diff_val)
    print(
        "   i = ",
        rmax_i,
        "  x[i] = ",
        rmax_x,
        "  y_h[i] = ",
        rmax_y_h,
        "  y_real = ",
        max_y_real,
    )


main()
