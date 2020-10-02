using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    public class BaseActivation : IActivationFunction<double> 
    {

        public virtual void SetAnswer(Matrix<double> answer)
        {
            throw new NotImplementedException();
        }

        public Vector<double> backPropagation(Vector<double> v)
        {
            return df(v);
        }

        public virtual Matrix<double> backPropagation(Matrix<double> m)
        {
            return Matrix<double>.Build.DenseOfRowVectors(m.EnumerateRows().Select(df));
        }

        public Vector<double> call(Vector<double> v)
        {
            return f(v);
        }

        public virtual Matrix<double> call(Matrix<double> m)
        {
            return Matrix<double>.Build.DenseOfRowVectors(m.EnumerateRows().Select(f));
        }

        protected virtual Vector<double> f(Vector<double> x)
        {
            throw new NotImplementedException();
        }

        protected virtual Vector<double> df(Vector<double> x)
        {
            throw new NotImplementedException();
        }

        override public string ToString()
        {
            return GetType().ToString().Split('.').Last();
        }
    }
}
