import matplotlib.pyplot as plt
import numpy


def minOneInPowK(k):
    return -1 if k % 2 else 1


def calcF(k, kFact, xInPowK):
    return minOneInPowK(k) * xInPowK / kFact


def calcPoint(eps, x):
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
        val = calcF(k, kFact, xInPowK)
    return (summ, k)


e = 0.1
h = .1
a = -2
b = 30

xArr = [i for i in numpy.arange(a, b, h)]
resArr = list(zip(*[calcPoint(e, i) for i in xArr]))
yArr = list(resArr[0])
kArr = list(resArr[1])

plt.grid()
plt.plot(xArr, yArr)
plt.plot(xArr, kArr)
plt.show()
