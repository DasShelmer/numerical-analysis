from math import sin


def f(x) -> float:
    return x**2


def integralTrapezoidal(a, b, n):
    h = (b - a) / n  # отрезок будет [a,b-h]
    summ = h * (f(a) + f(b)) / 2.0
    for i in range(n):
        summ += h * f(a + h * i)
    return summ


def integralSimpson(a, b, n):
    h = (b - a) / n  # отрезок будет [a,b-h]
    summ = 0.
    for i in range(n):
        x1 = h * i + a
        x2 = h * (i+1) + a
        summ += (x2-x1) / 6.0 * (f(x1) + 4.0*f((x1+x2)/2) + f(x2))
    return summ


def runge(fn, a, b, n, e):
    first = fn(a, b, n)
    second = fn(a, b, 2 * n)
    if abs(first - second) >= e:
        n *= 2
    while abs(first - second) >= e:
        n *= 2
        first = second
        second = fn(a, b, n)
    return second, n


def run():
    a = 0.
    b = 1.
    n = 1
    e = 1e-3
    print(runge(integralTrapezoidal, a, b, n, e * 3))
    print(runge(integralSimpson, a, b, n, e * 15))

run()
