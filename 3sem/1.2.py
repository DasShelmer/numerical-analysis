import matplotlib.pyplot as plt
from math import sin


# Возвращает N точек из отрезка [start, end]
def linspace(start, end, n):
    h = (end-start) / (n-1)
    return [h*a for a in range(n)]


def show_table(xArray, *yArrays):
    for i in range(len(xArray)):
        precision = '15.3f'
        X = format(xArray[i], precision)
        Ys = ' | '.join(format(yArray[i], precision) for yArray in yArrays)
        print(f'{X} | {Ys}')


def show_graph(xArray, *yArrays):
    plt.grid()
    for yArray in yArrays:
        plt.plot(xArray, yArray)
    plt.show()


def series(eps, x):
    k = 0
    kFact = 1
    xInPowK = 1

    val = 1
    summ = 0
    while (abs(val) > eps):
        summ += val
        k += 1
        kFact *= k
        xInPowK *= x
        val = (-1 if k % 2 else 1) * xInPowK / kFact
    return summ


def calc_series(a=0.0, b=5.0, n=10, e=0.1):
    xArr = [i for i in linspace(a, b, n)]
    yArr = [pow(x, 11) for x in xArr]

    return (xArr, yArr)


def lagrange(x, xArr, yArr):
    k = len(xArr)

    def calc_basis(j):
        polynom = 1
        for m in range(k):
            if m != j:
                polynom *= (x - xArr[m])/(xArr[j] - xArr[m])
        return polynom

    summ = 0
    for j in range(k):
        summ += calc_basis(j) * yArr[j]
    return summ


def calc_lagrange(xCalculated: list, yCalculated: list, n):
    assert len(xCalculated) >= 1 and len(xCalculated) == len(yCalculated)
    a = xCalculated[0]
    b = xCalculated[-1]

    xArrLan = linspace(a, b, n)
    yArrLan = [lagrange(x, xCalculated, yCalculated) for x in xArrLan]
    return yArrLan


def run():
    a = 0
    b = 5
    n = 21
    lagrange_n = 11
    xCalculated, yCalculated = calc_series(a, b, n)

    xForLagrange, yForLagrange = calc_series(a, b, lagrange_n)

    # Считаем и промежуточные и известные узлы
    yLagrange = calc_lagrange(xForLagrange, yForLagrange, n)
    yDelta = [abs(c - l) for c, l in zip(yCalculated, yLagrange)]

    show_table(xCalculated, yCalculated, yLagrange, yDelta)
    #show_graph(xCalculated, yCalculated, yLagrange)

run()
