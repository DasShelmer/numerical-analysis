import matplotlib.pyplot as plt
from math import sin
from decimal import Decimal
TYPE = Decimal

# Возвращает N точек из отрезка [start, end]
def linspace(start, end, n):
    h = (end-start) / (n-1)
    return [TYPE(h*a+start) for a in range(n)]


def show_table(xArray, *yArrays):
    for i in range(len(xArray)):
        precision = '15.3f'
        X = format(xArray[i], precision)
        Ys = ' | '.join(format(yArray[i], precision) for yArray in yArrays)
        print(f'{X} | {Ys}')


plt.grid()

def lagrange(x, xArr, yArr, plots):
    k = len(xArr)

    def calc_basis(j, x):
        polynom = TYPE(1)
        for m in range(k):
            if m != j:
                polynom *= (x - xArr[m])/(xArr[j] - xArr[m])
        return polynom
    summ = 0
    for j in range(k):
        if not j in plots:
            plots[j] = [calc_basis(j, xi) * yArr[j] for xi in linspace(-10, 10, 512)]
        summ += calc_basis(j, x) * yArr[j]
    return summ


def run():
    a = 0
    b = 3
    n = 15
    al = a
    bl = b
    nl = 10
    f = lambda x: x**9
    xCalculated = linspace(a, b, n)
    yCalculated = [f(x) for x in xCalculated]

    xForLagrange = linspace(al, bl, nl)
    yForLagrange = [f(x) for x in xForLagrange]

    plots = {}
    # Считаем и промежуточные и известные узлы
    xLagrange = linspace(a, b, n)
    yLagrange = [lagrange(x, xForLagrange, yForLagrange, plots) for x in xLagrange]
    yDelta = [abs(c - l) for c, l in zip(yCalculated, yLagrange)]
    maxDelta = max(yDelta)
    minDelta = min(yDelta)
    avrDelta = sum(yDelta) / len(yDelta)

    for p in plots.values():
        while len(p) < len(xLagrange):
            p.append(p[-1])
        plt.plot(linspace(-10, 10, 512), p)
    show_table(xCalculated[:10], yCalculated[:10], yLagrange[:10], yDelta[:10])
    plt.plot(xCalculated, yCalculated, 'ro', color='blue')
    plt.plot(xLagrange, yLagrange, color='red')
    print({'maxD': maxDelta, 'minD': minDelta, 'avrD':avrDelta})
    plt.show()


run()
