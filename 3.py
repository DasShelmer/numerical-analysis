from math import sin


def f(x) -> float:
    return sin(x)* x**3 / 2**x


def integral(a, b, n):
    h = (b - a) / n  # отрезок будет [a,b-h]
    summ = 0.
    for i in range(n):
        x1 = h * i + a
        x2 = h * (i+1) + a
        summ += (x2-x1) / 6.0 * (f(x1) + 4.0*f((x1+x2)/2) + f(x2))
    return summ


def runge(a, b, n, e=1e-3):
    eps = e / 15  # точность для I Симпсона 1/15
    first = integral(a, b, n)
    second = integral(a, b, 2 * n)
    while abs(first - second) / 15 >= eps:
        n *= 2
        first = second
        second = integral(a, b, n)
    return second, n

print(runge(0., 10., 1000))
