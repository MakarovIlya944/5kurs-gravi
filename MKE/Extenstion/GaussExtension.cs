using System;
using MKE.Matrix;

namespace MKE.Extenstion
{
    public static class GaussExtension
    {
        public static double[] Solve(DenseMatrix a, double[] f)
        {
            const double eps = 0.00001; // точность
            var x = new double[a.RowsCount];
            var k = 0;

            while (k < a.RowsCount)
            {
                // Поиск строки с максимальным a[i][k]
                var max = Math.Abs(a[k, k]);
                var index = k;

                for (var i = k + 1; i < a.RowsCount; i++)
                {
                    if (!(Math.Abs(a[i, k]) > max))
                    {
                        continue;
                    }

                    max = Math.Abs(a[i, k]);
                    index = i;
                }

                // Перестановка строк
                if (max < double.Epsilon)
                {
                    // нет ненулевых диагональных элементов
                    throw new Exception("Not found not zero diagonal element");
                }

                for (var j = 0; j < a.RowsCount; j++)
                {
                    var temp1 = a[k, j];
                    a[k, j] = a[index, j];
                    a[index, j] = temp1;
                }

                var temp = f[k];
                f[k] = f[index];
                f[index] = temp;

                // Нормализация уравнений
                for (var i = k; i < a.RowsCount; i++)
                {
                    var temp1 = a[i, k];

                    if (Math.Abs(temp1) < eps) continue; // для нулевого коэффициента пропустить

                    for (var j = 0; j < a.RowsCount; j++)
                        a[i, j] = a[i, j] / temp1;

                    f[i] = f[i] / temp1;

                    if (i == k) continue; // уравнение не вычитать само из себя

                    for (var j = 0; j < a.RowsCount; j++)
                        a[i, j] = a[i, j] - a[k, j];

                    f[i] = f[i] - f[k];
                }

                k++;
            }

            // обратная подстановка
            for (k = a.RowsCount - 1; k >= 0; k--)
            {
                x[k] = f[k];

                for (var i = 0; i < k; i++)
                    f[i] = f[i] - a[i, k] * x[k];
            }

            return x;
        }
    }
}