from numpy import abs, cos, sin, sqrt


def kramer(A, B):
    # Метод Крамера для решения СЛАУ 2 порядка
    det = A[0][0]*A[1][1] - A[1][0]*A[0][1]
    x0 = (B[0]*A[1][1] - B[1]*A[0][1]) / det
    x1 = (B[1]*A[0][0] - B[0]*A[1][0]) / det
    return [x0, x1]


def norm(x):
    # Подсчёт нормы вектора
    res = 0
    for i in range(len(x)):
        res += x[i] * x[i]
    return sqrt(res)


"""
/ x^2 + y^2 = 4
|
\ y = sin(x)
"""


def system(x):
    # Данная система нелинейных уравнений
    return [
        x[0]*x[0] + x[1]*x[1] - 4,
        x[1] - sin(x[0])
    ]


def jacobi(x):
    # Якобиан данной системы уравнений
    return [
        [2*x[0], 2*x[1]],
        [-cos(x[0]), 1]
    ]


def main():
    # Требуемая точность
    eps = 1e-1
    # Ограничение кол-ва итераций
    maxiter = 25
    # Начальное приближение
    xk = [0, 0]

    norm_x = 0
    norm_xk = 0
    iteration = 0

    print("Итерация: ", iteration)
    print("X = ", ' '.join([str(x) for x in xk]))

    norm_x = norm_xk  # Норма для предыдущего вектора

    # Рассчитываем значения якобиана и системы ур.
    a = jacobi(xk)
    b = system(xk)

    # Решаем СЛАУ второго порядка
    p = kramer(a, b)

    # Приближаем
    for i in range(len(xk)):
        xk[i] = xk[i] - p[i]
    norm_xk = norm(xk)  # Норма для нового вектора

    iteration += 1

    # Условие продолжения выполения итераций
    while abs(norm_x - norm_xk) >= eps and iteration < maxiter:
        print('------------------------')
        print("Итерация: ", iteration)
        print("Погрешность: ", abs(norm_x - norm_xk))
        print("X = ", ' '.join([str(x) for x in xk]))

        norm_x = norm_xk  # Норма для предыдущего вектора

        # Рассчитываем значения якобиана и системы ур.
        a = jacobi(xk)
        b = system(xk)

        # Решаем СЛАУ
        sole = SOLEGauss(a, b)
        sole.forward_run()
        p = sole.answer()
        # Приближаем
        for i in range(len(xk)):
            xk[i] = xk[i] - p[i]
        norm_xk = norm(xk)  # Норма для нового вектора
        iteration += 1

    print('------------------------')
    print('Решение СУ:')
    print('X = ', ' '.join([str(x) for x in xk]))
    print("Погрешность: ", abs(norm_x - norm_xk))
    print('За ', iteration, ' итераций')


if __name__ == '__main__':
    main()
