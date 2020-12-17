using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;
using System.Linq.Expressions;

namespace Mnist.Functions
{
    public class BaseLoss : ILossFunction<double>
    {
        public virtual double call(Vector<double> calc, Vector<double> truly)
        {
            throw new NotImplementedException();
        }

        public virtual Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            throw new NotImplementedException();
        }

        public Vector<double> call(Matrix<double> calc, Matrix<double> truly)
        {
            var t = truly.EnumerateRows();
            return Vector<double>.Build.DenseOfEnumerable(calc.EnumerateRows().Select((x, i) => call(x, t.ElementAt(i))));
        }

        public Matrix<double> backPropagation(Matrix<double> calc, Matrix<double> truly)
        {
            var t = truly.EnumerateRows();
            return Matrix<double>.Build.DenseOfRowVectors(calc.EnumerateRows().Select((x, i) => backPropagation(x, t.ElementAt(i))));
        }
    }
}