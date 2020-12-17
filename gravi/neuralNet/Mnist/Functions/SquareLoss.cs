using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    /*
    Loss fuction:
    sum((y-x)^2)/2

    Derivative:
    -(y-x)
    */

    class SquareLoss : ILossFunction<double>
    {
        public double call(Vector<double> calc, Vector<double> truly)
        {
            return (truly - calc).Map(x => x * x).Sum() / 2;
        }

        public Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            return -(truly - calc);
        }

        public Vector<double> call(Matrix<double> calc, Matrix<double> truly)
        {
            var t = truly.EnumerateRows();
            return Vector<double>.Build.DenseOfEnumerable(calc.EnumerateRows().Select((x, i) => (t.ElementAt(i) - x).Map(x => x * x).Sum() / 2));
        }

        public Matrix<double> backPropagation(Matrix<double> calc, Matrix<double> truly)
        {
            return -(truly - calc);
        }
    }
}