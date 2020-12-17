using System;
using System.Collections.Generic;
using System.Net.Cache;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    /*
    Loss fuction:
    sqrt(sum((y-x)^2))

    Derivative:
    ?
    */

    class L2Loss : BaseLoss
    {
        public override Vector<double> backPropagation(Vector<double> calc, Vector<double> truly)
        {
            double l2norm = call(calc, truly);
            if (Math.Abs(l2norm) < 1E-15)
                return Vector<double>.Build.Dense(calc.Count, 0);
            return (calc - truly) / l2norm;
        }

        public override double call(Vector<double> calc, Vector<double> truly)
        {
            return (calc - truly).L2Norm();
        }
    }
}
 