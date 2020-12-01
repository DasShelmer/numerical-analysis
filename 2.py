from typing import Callable
import matplotlib.pyplot as plt
from math import cos, sin


# Возвращает N точек из отрезка [start, end]
def linspace(start, end, n):
    h = (end-start) / (n-1)
    return [h*a for a in range(n)]


def show_table(xArray, *yArrays):
    for i in range(len(xArray)):
        precision = '10.3f'
        X = format(xArray[i], precision)
        Ys = ' | '.join(format(yArray[i], precision) for yArray in yArrays)
        print(f'{X} | {Ys}')


def show_graph(xArray, *yArrays):
    plt.grid()
    for yArray in yArrays:
        plt.plot(xArray, yArray)
    plt.show()


def f(x):
    return pow(cos(x), 2)


def f_diff(x):
    return -sin(2*x)


def diff(func: Callable, a, b, n):
    h = (b-a) / (n-1)
    return [(func(h*i+h)-func(h*i-h))/(2*h) for i in range(n)]


def run():
    n = 30
    a = -1
    b = 6
    xArray = linspace(a, b, n)
    difArray = diff(f, a, b, n)
    trueDifArray = [f_diff(x) for x in xArray]
    deltaArray = [d-td for td, d in zip(trueDifArray, difArray)]

    show_table(xArray, trueDifArray, difArray, deltaArray)
    show_graph(xArray, trueDifArray, difArray)


run()
