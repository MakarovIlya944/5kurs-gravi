using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.Text;

namespace Mnist
{
    public interface ILayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        public Vector<T> forward(Vector<T> input);
        public Matrix<T> forward(Matrix<T> input);
        public Vector<T> backPropagation(Vector<T> input, double rate);
        public Matrix<T> backPropagation(Matrix<T> prevW, Matrix<T> input, double rate, out Matrix<T> derivative);
    }
}
