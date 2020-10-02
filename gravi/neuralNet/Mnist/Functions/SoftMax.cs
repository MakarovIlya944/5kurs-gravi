using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace Mnist.Functions
{
    public class SoftMax : BaseActivation
    {
        int dim = 10;
        Vector<double> shifted;
        public int[] rightIndecies;

        public SoftMax(int dim)
        {
            this.dim = dim;
        }

        public SoftMax()
        {
        }

        public override void SetAnswer(Matrix<double> answer)
        {
            rightIndecies = answer.EnumerateRows().Select(x => x.Find(v => Math.Abs(1 - v) < 1E-15).Item1).ToArray();
        }

        public override Matrix<double> backPropagation(Matrix<double> m)
        {
            return Matrix<double>.Build.DenseOfRowVectors(m.EnumerateRows().Select((v,i) => df(v,i)));
        }

        protected override Vector<double> f(Vector<double> x)
        {
            shifted = (x - x.Max()).PointwiseExp();
            return shifted / shifted.Sum();
        }

        private Vector<double> df(Vector<double> x, int index)
        {
            shifted = f(x);
            int sIndex = rightIndecies[index];
            double sIndexed = shifted[sIndex];
            return shifted.MapIndexed((i, v) => i == sIndex ? (1 - v) * sIndexed : -v * sIndexed);
        }

        override public string ToString()
        {
            return base.ToString() + $" {dim}";
        }
    }
}
