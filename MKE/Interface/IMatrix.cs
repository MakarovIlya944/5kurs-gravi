using System;
using System.Text;

namespace MKE.Interface
{
    public interface IMatrix<T> where T : unmanaged
    {
        void MultiplyInverse(T[] x);
        void Multiply(T[] x, T[] y, T? mult);
        void MultiplyT(T[] x, T[] y, T? mult);
        void Lx(T[] x);
        void Ux(T[] x);
        int Rows { get; }
        int Columns { get; }
        T this[int i, int j] { get; set; }
        void ThreadSafeAdd(int i, int j, T value);
        void ThreadSafeSet(int i, int j, T value);
        void AddTo(Action<int, int, T> add);
        void SetTo(Func<int, int, T> value);
        void Preconditioner(T[] x, T[] y);
    }
}