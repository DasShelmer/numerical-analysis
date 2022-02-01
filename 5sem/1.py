from math import *

import numpy as np


def f1(x, y):
    # Диффур (1)
    # y' = 3x^2 +2x + cos(x)(x^3+x^2+x - y) + 1
    return 3 * x ** 2 + 2 * x + cos(x) * (x ** 3 + x ** 2 + x - y) + 1


def y1(x, C):
    # Ответ для (1)
    # y = C e^(-sin(x)) + x^3 + x^2 + x
    return C * exp(-sin(x)) + x ** 3 + x ** 2 + x


f_ = f1
y_ = y1


def RK4(a: float, b: float, n: int, y0: float):
    Y = np.zeros(n + 1)  # n + 1 нулей
    Y[0] = y0

    h = (b - a) / n
    x = a
    for i in range(n):
        K1 = f_(x, Y[i])
        K2 = f_(x + h / 2.0, Y[i] + K1 * h / 2.0)
        K3 = f_(x + h / 2.0, Y[i] + K2 * h / 2.0)
        K4 = f_(x + h, Y[i] + K3 * h)
        Y[i + 1] = Y[i] + (h / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)

        x += h

    return Y


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


def fshow(T, T_, n, a, b, C):
    # Красивый вывод таблички

    offset = 10
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
        y_real = y_(x_i, C)
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
    eps = 1e-6

    a = 0.0
    b = 2.0
    N = 4
    C = 0.0

    y0 = y_(a, C)

    n = N
    subdiv = 0
    T = RK4(a, b, n, y0)
    n = 2 * n
    subdiv += 1
    T_ = RK4(a, b, n, y0)

    while max_diff(T, T_)[1] > eps:
        T = T_
        n = 2 * n
        subdiv += 1
        T_ = RK4(a, b, n, y0)

    fshow(T, T_, n + 1, a, b, C)

    T_real = [y_(a + i * (b - a) / n, C) for i in range(n)]

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
