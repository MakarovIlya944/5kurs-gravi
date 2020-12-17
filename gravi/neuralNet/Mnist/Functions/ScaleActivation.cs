using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;

namespace Mnist.Functions
{
    class ScaleActivation : BaseActivation
    {
        protected override Vector<double> f(Vector<double> x)
        {
            double max = x.Maximum();
            return x.Map(e => e > 0 ? e / max : 0);
        }

        protected override Vector<double> df(Vector<double> x)
        {
            double max = x.Maximum();
            return x.Map(e => e > 0 ? 1 / max : 0);
        }
    }
}
