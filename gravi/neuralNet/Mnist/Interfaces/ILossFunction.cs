using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist
{
    public interface ILossFunction<T> where T : struct, IEquatable<T>, IFormattable
    {
        public T call(Vector<T> calc, Vector<T> truly);
        public Vector<T> backPropagation(Vector<T> calc, Vector<T> truly);

        public Vector<T> call(Matrix<T> calc, Matrix<T> truly);
        public Matrix<T> backPropagation(Matrix<T> calc, Matrix<T> truly);
    }
}
