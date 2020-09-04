using System;
using System.Linq;
using MKE.Interface;
using MKE.Matrix;
using MKE.Utils;

namespace MKE.Solver {
    public class CGSolver:ISolver<double>
    {
        public long MaxIteration = 1000;

        public double Eps = 1e-15;

        public FactorizationType FactorizationType;

        public void Initialization(long maxIteration, double eps, FactorizationType factorization)
        {
            MaxIteration = maxIteration;
            Eps = eps;
            FactorizationType = factorization;
        }

        public double[] Solve(IMatrix<double> Matrix, double[] b, double[] startX)
        {
            var LA = LinearAlgebra.Get<double>();
            var matrix = Factorization.Factorize<double>(FactorizationType, Matrix); //  lu();
            var bNorm = LA.Norm(b);

            if (bNorm == 0)
                return b;

            int n = b.Length;
            var r = new double[n];
            var z = new double[n];
            var p = new double[n];
            var x = new double[n];
            var t = new double[n];

            b.ToList().CopyTo(r);

            Matrix.Multiply(startX, p, 1);
            LA.Sub(r, LA.Cast(1), p);
            startX.ToList().CopyTo(x);

            matrix.Preconditioner(r, z);
            z.ToList().CopyTo(p);
            var rz = LA.Dot(r, z);

            for (int i = 1; true; ++i)
            {
                Matrix.Multiply(p, t, 1);
                var alpha = LA.Div(rz, LA.Dot(t, p));

                LA.Add(x, alpha, p);
                LA.Sub(r, alpha, t);

                var rNorm = LA.Norm(r);

                if (rNorm < Eps * bNorm)
                {
                    return x;
                }

                if (i == MaxIteration)
                {
                    throw new Exception($"Failed: iter {i} (max = {MaxIteration}), residual {rNorm / bNorm} (eps {Eps})");
                }

                matrix.Preconditioner(r, z);
                var r1z = LA.Dot(r, z);
                var beta = LA.Div(r1z, rz);
                rz = r1z;

                z.ToList().CopyTo(t);
                LA.Add(t, beta, p);
                t.ToList().CopyTo(p);
            }
        }
    }
}