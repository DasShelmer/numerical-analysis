# Метод Гаусса для решения СЛАУ с выбором глав эл-та
# и проверкой на правильность ответа
import numpy as np


class SOLEGauss:
    def __init__(self, A, B):
        self.A = [[float(e) for e in row] for row in A]
        self.B = [float(e) for e in B]

    def print(self, selected=(-1, -1)):
        # Вывод матрицы на экран
        for row in range(len(self.A)):
            for col in range(len(self.A[0])):
                item = "%5.2f" % self.A[row][col]
                if selected == (row, col):
                    item += "*"
                else:
                    item += " "
                print(item, end="")
            print("|", "%5.2f" % self.B[row])
        print()

    def swap_rows(self, row1, row2):
        # Смена строк row1 и row2 местами
        self.A[row1], self.A[row2] = self.A[row2], self.A[row1]
        self.B[row1], self.B[row2] = self.B[row2], self.B[row1]

    def divide_row(self, row, divider):
        # Деление строки row на divider
        self.A[row] = [e / divider for e in self.A[row]]
        self.B[row] /= divider

    def combine_rows(self, changed_row, row, multiplier):
        # Сложение строк row += multiplier * src_row
        self.A[changed_row] = [
            (e1 + multiplier * e2) for e1, e2 in zip(self.A[changed_row], self.A[row])
        ]
        self.B[changed_row] += multiplier * self.B[row]

    def forward_run(self):
        # Приведение матрицы к треугольному виду
        for col in range(len(self.B)):
            max_row = None
            for row in range(col, len(self.A)):
                if max_row is None or abs(self.A[row][col]) > abs(self.A[max_row][col]):
                    max_row = row
            # print('Ищем макс. эл. в %d-м столбце:' % (col + 1))
            # self.print((max_row, col))
            if max_row != col:
                self.swap_rows(max_row, col)
                # print('Ставим строку с макс. эл. выше:')
                # self.print((col, col))

            if np.abs(self.A[col][col]) > np.finfo(float).eps:
                self.divide_row(col, self.A[col][col])
                # print('Нормализуем строку с макс. эл.:')
                # self.print((col, col))

            for row in range(col + 1, len(self.A)):
                self.combine_rows(row, col, -self.A[row][col])

            # print('Обрабатываем строки снизу:')
            # self.print((col, col))
        # print('Матрица приведена к треугольному виду')

    def answer(self):
        # Получение вектора X (обратный проход)
        X = [0] * len(self.B)
        for row in range(len(self.B) - 1, -1, -1):
            row_sum = sum(x * a for x, a in zip(X, self.A[row]))
            X[row] = self.B[row] - row_sum

        return X

    def is_correct_answer(self, X):
        # Проверка на правильность ответа
        for row in range(len(self.B) - 1, -1, -1):
            row_sum = sum(a * x for x, a in zip(X, self.A[row]))
            if self.B[row] != row_sum:
                return False
        return True


def main():
    A = [[1, 2, 3], [4, 5, 6], [1, 0, 1]]
    B = [1, 1, 1]

    sole = SOLEGauss(A, B)
    print("Введёная СЛАУ:")
    sole.print()

    sole.forward_run()
    print("Треугольный вид:")
    sole.print()

    X = sole.answer()
    print("Ответ:")
    print(", ".join(["x%d=%5.2f" % (i, X[i]) for i in range(len(X))]))

    is_correct = "Да, верный" if sole.is_correct_answer(X) else "Нет, неверный"
    print("Верный ли ответ:", is_correct)


if __name__ == "__main__":
    main()
