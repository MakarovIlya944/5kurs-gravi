using System;
using System.Collections.Generic;
using MKE.Extenstion;
using MKE.Interface;

namespace MKE.Matrix
{
    public class SparseMatrixReal : SparseMatrix<double>
    {
        public SparseMatrixReal(double[] ggl, double[] ggu, int[] ig, int[] jg, double[] di) : base(ggl, ggu, ig, jg, di) { }
        public SparseMatrixReal(IList<IEnumerable<int>> portrait):base(portrait)
        {
           
        }
        protected SparseMatrixReal() : base()
        {
        }
        protected SparseMatrixReal(SparseMatrix<double> matrix) : base(matrix)
        {
        }

        public override void Lx(double[] y)
        {
            for (int k = 0; k < Size; k++)
            {
                double s = 0.0;
                for (int i = ig[k]; i < ig[k + 1]; i++)
                {
                    int j = jg[i];
                    s += ggl[i] * y[j];
                }

                y[k] = (y[k] - s) / di[k];
            }
        }

        public override void Ux(double[] y)
        {
            for (int k = Size - 1; k >= 0; k--)
            {
                double yk = y[k];
                for (int j = ig[k]; j < ig[k + 1]; j++)
                {
                    int i = jg[j];
                    y[i] -= ggu[j] * yk;
                }
            }
        }

        public override SparseMatrix<double> LUsqFactorize()
        {
            var M = new SparseMatrixReal(this);
            for (int i = 0; i < Size; i++)
            {
                M.di[i] = di[i];
                for (int l = ig[i]; l < ig[i + 1]; l++)
                {
                    int j = jg[l];
                    int a = ig[i], b = ig[j];
                    double tl = 0, tu = 0;
                    while (a < l &&
                           b < ig[j + 1])
                        //while (a < ig[i + 1] && b < ig[j + 1])
                    {
                        if (jg[a] == jg[b])
                        {
                            tl += M.ggl[a] * M.ggu[b];
                            tu += M.ggl[b++] * M.ggu[a++];
                        }
                        else if (jg[a] < jg[b]) a++;
                        else b++;
                    }

                    M.ggl[l] = (ggl[l] - tl) / M.di[j];
                    M.ggu[l] = (ggu[l] - tu) / M.di[j];
                    M.di[i] -= M.ggl[l] * M.ggu[l];
                }

                //if (M.di[i] < 1e-20)
                //{
                //    progress.Adddoubleext("LUsqFactorize error");
                //    return new DiagonalMatrix(di);
                //}
                M.di[i] = Math.Sqrt(M.di[i]);

                //progress.Increment();
            }

            return M;
        }

        public override void ThreadSafeAdd(int i, int j, double value)
        {
            if (i > j)
                ggl.ThreadSafeAdd(Array.BinarySearch(jg, ig[i], ig[i + 1] - ig[i], j), value);
            else if (i < j)
                ggu.ThreadSafeAdd(Array.BinarySearch(jg, ig[j], ig[j + 1] - ig[j], i), value);
            else
                di.ThreadSafeAdd(i, value);
        }
        public override void ThreadSafeSet(int i, int j, double value)
        {
            if (i > j)
                ggl.ThreadSafeSet(Array.BinarySearch(jg, ig[i], ig[i + 1] - ig[i], j), value);
            else if (i < j)
                ggu.ThreadSafeSet(Array.BinarySearch(jg, ig[j], ig[j + 1] - ig[j], i), value);
            else
                di.ThreadSafeSet(i, value);
        }

        public override void MultiplyInverse(double[] x)
        {
            for (int i = 0; i < Size; i++)
            {
                for (int j = ig[i]; j < ig[i + 1]; j++)
                    x[i] -= ggl[j] * x[jg[j]] / di[jg[j]];
            }

            for (int i = 0; i < Size; i++) x[i] /= di[i];
            for (int i = Size - 1; i >= 0; i--)
            {
                double a = x[i] / di[i];
                for (int j = ig[i]; j < ig[i + 1]; j++)
                    x[jg[j]] -= a * ggu[j];
            }

            for (int i = 0; i < Size; i++) x[i] /= di[i];
        }

        public override void Multiply(double[] x, double[] y, double? mult)
        {
            for (int i = 0; i < _size; i++)
            {
                y[i] = di[i] * x[i];
                for (int j = ig[i]; j < ig[i + 1]; j++)
                {
                    y[i] += ggl[j] * x[jg[j]];
                    y[jg[j]] += ggu[j] * x[i];
                }
            }
        }

        public override void MultiplyT(double[] x, double[] y, double? mult)
        {
            for (int i = 0; i < _size; i++)
            {
                y[i] = di[i] * x[i];
                for (int j = ig[i]; j < ig[i + 1]; j++)
                {
                    y[i] += ggu[j] * x[jg[j]];
                    y[jg[j]] += ggl[j] * x[i];
                }
            }

            for (int i = 0; i < _size; i++)
            {
                y[i] *= mult.Value;
            }
        }
    }
}