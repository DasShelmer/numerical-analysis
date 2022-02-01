import numpy as np

from gauss import SOLEGauss


def kernel(x, s):
    return np.sin(x) * np.cos(s)


y_x = lambda x: np.cos(2 * x)

f_x = lambda x: np.cos(2 * x)


def kernel(x, s):
    return np.sin(x) + s


y_x = lambda x: x * np.sin(2 * x)

f_x = lambda x: -np.pi * (np.sin(x) + np.pi) / 2


def simpson(xn, h):
    n = len(xn) - 1
    matrix = [[1.0 if i == j else 0.0 for i in range(n + 1)] for j in range(n + 1)]
    for i in range(n + 1):
        for j in range(1, n, 2):
            matrix[i][j - 1] += kernel(xn[i], xn[j - 1]) * 2 * h / 6
            matrix[i][j] += kernel(xn[i], xn[j]) * 8 * h / 6
            matrix[i][j + 1] += kernel(xn[i], xn[j + 1]) * 2 * h / 6
    return matrix


def quadrature_simpson(a, b, n):
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]  # [a, a+h, a+2h, ..., b-h, b]

    matrix_simpson = simpson(x, h)
    f_xn = list(map(f_x, x))

    sole = SOLEGauss(matrix_simpson, f_xn)
    sole.forward_run()
    result = sole.answer()
    return result


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

    offset = n // 5
    maxoffset = 10
    if offset > maxoffset:
        offset = maxoffset

    indexes = [
        *range(offset),  # Начало
        -1,
        *range(n // 2 - offset // 2, n // 2 + offset // 2),  # Середина
        -1,
        *range(n - (offset - 1), n),  # Конец
    ]
    indexes = range(n)  # Всё

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
    eps = 1e-3

    a = 0.0
    b = 2 * np.pi
    N = 10

    n = N
    subdiv = 0
    T = quadrature_simpson(a, b, n)
    n = 2 * n
    subdiv += 1
    T_ = quadrature_simpson(a, b, n)

    while max_diff(T, T_)[1] > eps:
        T = T_
        n = 2 * n
        subdiv += 1
        T_ = quadrature_simpson(a, b, n)

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
