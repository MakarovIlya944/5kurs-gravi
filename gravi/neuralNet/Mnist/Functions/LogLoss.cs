using System;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    /*
    Loss fuction:
    -sum(x*log(y))

    Derivative:
    -(y-x)
    */

    class LogLoss : BaseLoss
    {
        public override double call(Vector<double> calc, Vector<double> truly)
        {
            int trueClass = truly.Find(v => Math.Abs(v - 1) < 1E-15).Item1; // only for data with one 1 and all other 0
            //if (double.IsInfinity(Math.Log(calc[trueClass])))
            //    return -1E+100;
            //else
                return -Math.Log(calc[trueClass]);
        }

        public override Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            return -(truly.PointwiseDivide(calc));
        }
    }
}
 