using System;
using System.Collections.Generic;
using System.Linq;
using MKE.Interface;

namespace MKE.Matrix
{
    public class SparseMatrix<T> : IMatrix<T> where T : unmanaged
    {
        protected int _size;
        protected int[] ig;
        protected int[] jg;
        protected T[] di;
        protected T[] ggl;
        protected T[] ggu;

        public int Rows
        {
            get { return _size; }
        }

        public int Columns
        {
            get { return _size; }
        }

        public IEnumerable<int> RowIndices
        {
            get { return Enumerable.Range(0, Rows); }
        }

        public IEnumerable<int> ColumnIndices
        {
            get { return Enumerable.Range(0, Columns); }
        }

        public int Size
        {
            get { return _size; }
        }

        public SparseMatrix(T[] ggl, T[] ggu, int[] ig, int[] jg, T[] di)
        {
            this.ggl = ggl;
            this.ggu = ggl;
            this.di = di;
            this.ig = ig;
            this.jg = jg;
            _size = di.Length;
        }
        public SparseMatrix(IList<IEnumerable<int>> portrait)
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
                    jg[k++] = cnum;
            }

            di = new T[_size];
            ggl = new T[ig[_size]];
            ggu = new T[ig[_size]];
        }

        public virtual long MemorySize => (ig.Length + jg.Length) * sizeof(int) +
                                          (di.Length + ggl.Length + ggu.Length) * sizeof(double);

        public virtual T this[int i, int j]
        {
            get
            {
                if (i > j)
                    return ggl[Array.BinarySearch(jg, ig[i], ig[i + 1] - ig[i], j)];
                if (i < j)
                    return ggu[Array.BinarySearch(jg, ig[j], ig[j + 1] - ig[j], i)];
                return di[i];
            }
            set
            {
                if (i > j)
                {
                    ggl[Array.BinarySearch(jg, ig[i], ig[i + 1] - ig[i], j)] = value;
                }
                else
                {
                    if (i < j)
                    {
                        ggu[Array.BinarySearch(jg, ig[j], ig[j + 1] - ig[j], i)] = value;
                    }
                    else
                    {
                        di[i] = value;
                    }
                }
            }
        }

        protected SparseMatrix()
        {
        }
        public void Preconditioner(T[] x, T[] y)
        {
            for (int i = 0; i < x.Length; i++)
            {
                y[i] = x[i];
            }
            MultiplyInverse(y);
        }
        protected SparseMatrix(SparseMatrix<T> matrix)
        {
            _size = matrix._size;
            ig = matrix.ig;
            jg = matrix.jg;
            di = new T[_size];
            ggl = new T[ig[_size]];
            ggu = new T[ig[_size]];
        }

        public virtual void Lx(T[] y)
        {
            throw new NotImplementedException();
        }

        public virtual void Ux(T[] y)
        {
            throw new NotImplementedException();
        }

        public virtual SparseMatrix<T> LUsqFactorize()
        {
            throw new NotImplementedException();
        }

        public virtual void ThreadSafeAdd(int i, int j, T value)
        {
            throw new NotImplementedException();
        }
        public virtual void ThreadSafeSet(int i, int j, T value)
        {
            throw new NotImplementedException();
        }
        public virtual void AddTo(Action<int, int, T> add)
        {
            for (int i = 0; i < _size; i++)
                add(i, i, di[i]);
            for (int i = 0; i < _size; i++)
            for (int j = ig[i]; j < ig[i + 1]; j++)
            {
                add(i, jg[j], ggl[j]);
                add(jg[j], i, ggu[j]);
            }
        }

        public virtual void SetTo(Func<int, int, T> value)
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
                    ggu[j] = value(jg[j], i);
                }
            }
        }

        public virtual void MultiplyInverse(T[] x)
        {
            throw new NotImplementedException();
        }

        public virtual void Multiply(T[] x, T[] y, T? mult)
        {
            throw new NotImplementedException();
        }

        public virtual void MultiplyT(T[] x, T[] y, T? mult)
        {
            throw new NotImplementedException();
        }
    }
}