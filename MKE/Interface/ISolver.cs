using System.Collections.Generic;
using System.Numerics;
using MKE.Matrix;

namespace MKE.Interface
{
    public interface ISolver<T> where T: unmanaged
    {
        void Initialization(long maxIteration, double eps, FactorizationType factorization);
        T[] Solve(IMatrix<T> Matrix, T[] b, T[] startX);
    }
}