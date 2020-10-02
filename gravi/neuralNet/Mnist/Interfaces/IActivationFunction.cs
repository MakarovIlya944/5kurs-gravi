using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist
{
    public interface IActivationFunction<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Vector<T> call(Vector<T> v);
        public void SetAnswer(Matrix<double> m);
        public Matrix<T> call(Matrix<T> v);
        public Vector<T> backPropagation(Vector<T> v);
        public Matrix<T> backPropagation(Matrix<T> v);
    }
}
