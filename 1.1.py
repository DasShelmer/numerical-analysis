import matplotlib.pyplot as plt
import numpy


def minOneInPowK(k):
    return -1 if k % 2 else 1


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
        val = minOneInPowK(k) * xInPowK / kFact
    return (summ, k)


e = 0.1
n = 10
a = 0
b = 5
h = (b - a)/n

xArr = [i for i in numpy.arange(a, b, h)]
resArr = list(zip(*[calcPoint(e, i) for i in xArr]))
yArr = list(resArr[0])
kArr = list(resArr[1])

print(xArr)
print(yArr)

plt.grid()
plt.plot(xArr, yArr)
plt.show()
