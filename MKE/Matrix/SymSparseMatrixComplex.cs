using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using MKE.Extenstion;
using MKE.Interface;

namespace MKE.Matrix
{
    public class SymSparseMatrixComplex : SparseMatrixComplex
    {
        public override long MemorySize =>
            (ig.Length + jg.Length) * sizeof(int) + (di.Length + ggl.Length) * sizeof(double);
        public SymSparseMatrixComplex(Complex[] gg, int[] ig, int[] jg, Complex[] di) : base(gg, gg, ig, jg, di) { }

        protected SymSparseMatrixComplex(SymSparseMatrixComplex matrix)
        {
            _size = matrix._size;
            ig = matrix.ig;
            jg = matrix.jg;
            di = new Complex[_size];
            ggl = ggu = new Complex[ig[_size]];
        }

        public SymSparseMatrixComplex(IList<IEnumerable<int>> portrait)
        {
            _size = portrait.Count;
            ig = new int[_size + 1];
            for (int i = 0; i < _size; i++)
                ig[i + 1] = ig[i] + portrait[i].Count(j => j < i);
            jg = new int[ig[_size]];
            int k = 0;
            for (int i = 0; i < _size; i++)
            {
                foreach (var cnum in portrait[i].Where(j => j < i))
                {
                    jg[k++] = cnum;
                }
            }

            di = new Complex[_size];
            ggl = ggu = new Complex[ig[_size]];
        }

        public override Complex this[int i, int j]
        {
            get { return base[i, j]; }
            set
            {
                if (i > j)
                {
                    ggl[Array.BinarySearch(jg, ig[i], ig[i + 1] - ig[i], j)] = value;
                }
                else
                {
                    if (i == j)
                    {
                        di[i] = value;
                    }
                }
            }
        }

        public override void SetTo(Func<int, int, Complex> value)
        {
            for (int i = 0; i < _size; i++)
            {
                di[i] = value(i, i);
            }

            for (int i = 0; i < _size; i++)
            {
                for (int j = ig[i]; j < ig[i + 1]; j++)
                {
                    ggl[j] = value(i, jg[j]);
                }
            }
        }

        public override void ThreadSafeAdd(int i, int j, Complex value)
        {
            if (i > j)
            {
                ggl.ThreadSafeAdd(Array.BinarySearch(jg, ig[i], ig[i + 1] - ig[i], j), value);
            }
            else
            {
                if (i == j)
                {
                    di.ThreadSafeAdd(i, value);
                }
            }
        }
        public override void ThreadSafeSet(int i, int j, Complex value)
        {
            if (i > j)
            {
                ggl.ThreadSafeSet(Array.BinarySearch(jg, ig[i], ig[i + 1] - ig[i], j), value);
            }
            else
            {
                if (i == j)
                {
                    di.ThreadSafeSet(i, value);
                }
            }
        }

        public SparseMatrix<Complex> LLtFactorize()
        {
            var M = new SymSparseMatrixComplex(this);
            for (int i = 0; i < Size; i++)
            {
                M.di[i] = di[i];
                for (int l = ig[i]; l < ig[i + 1]; l++)
                {
                    int j = jg[l];
                    int a = ig[i], b = ig[j];
                    Complex tl = 0;
                    while (a < l && b < ig[j + 1])
                    {
                        if (jg[a] == jg[b])
                        {
                            tl += M.ggl[a++] * M.ggl[b++];
                        }
                        else
                        {
                            if (jg[a] < jg[b])
                            {
                                a++;
                            }
                            else
                            {
                                b++;
                            }
                        }
                    }

                    M.ggl[l] = (ggl[l] - tl) / M.di[j];
                    M.di[i] -= M.ggl[l] * M.ggl[l];
                }

                //if (M.di[i] < 1e-20)
                //{
                //    progress.AddText("LLtFactorize error");
                //    return new DiagonalMatrix(di);
                //}
              //  M.di[i] = Complex.Sqrt(Math.Abs(M.di[i]));//TODO: whaaaaat
            }

            return M;
        }
    }
}