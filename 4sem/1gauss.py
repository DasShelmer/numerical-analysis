
class GaussSOLE:
    def __init__(self, A, B, selR, selC):
        self.selR = selR
        self.selC = selC
        self.A = A
        self.B = B

    def print(self):
        for rowI in range(0, len(self.A)):
            for colI in range(0, len(self.B)):
                print('%5.1f' % self.A[rowI][colI], end='')
            print('|', '%5.1f' % self.B[colI])

    def add_row(self, sr, ar, mul):
        # Добавляет к строке sr строку ar * mul
        self.A[sr] = [e1 + mul*e2 for e1, e2 in zip(self.A[sr], self.A[ar])]
        self.B[sr] += mul * self.B[ar]

    def swap_rows(self, r1, r2):
        # Меняет местами строки
        temp = self.A[r1]
        self.A[r1] = self.A[r2]
        self.A[r2] = temp

        temp = self.B[r1]
        self.B[r1] = self.B[r2]
        self.B[r2] = self.B[r1]

    def mul_row(self, r, mul):
        self.A[r] = [mul*el for el in self.A[r]]
        self.B[r] *= mul

    def to_triangle_view(self):
        column = 0
        while (column < len(self.B)):
            current_row = None
            for r in range(column, len(self.A)):
                if current_row is None or abs(self.A[r][column]) > abs(self.A[current_row][column]):
                    current_row = r

            if current_row is None:
                print('Нет решений')
                return

            if current_row != column:
                self.swap_rows(current_row, column)
                self.print()
            self.mul_row(column, 1.0/self.A[column][column])
            for r in range(column + 1, len(self.A)):
                self.add_row(r, column, -self.A[r][column])
            column += 1

    def answer(self):
        X = [0 for b in self.B]
        for i in range(len(self.B) - 1, -1, -1):
            X[i] = self.B[i] - \
                sum(x * a for x, a in zip(X[(i + 1):], self.A[i][(i + 1):]))

        return X


def main():
    A = [
        [2., 1., -1., 3.],
        [1., -1., 0., 1.],
        [1., 0., -1., 1.]
    ]
    A = [
        [2., 1., 1.],  # *x1
        [1., -1., 0.],  # *x2
        [-1., 0., -1.],  # *x3
        [3., 1., 1.]  # *x4
    ]
    B = [-2., 0., -2.]
    gsole = GaussSOLE(A, B, 1, 1)
    print('СЛАУ:')
    gsole.print()
    gsole.to_triangle_view()
    print('Треугольный вид СЛАУ:')
    gsole.print()
    print('Ответ:')
    print(gsole.answer())


main()
