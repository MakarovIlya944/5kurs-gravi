using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using MKE.Extenstion;
using MKE.Interface;
using MKE.Matrix;

namespace MKE.Solver
{
    public class ComplexLos:ISolver<Complex>
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

        public Complex[] Solve(IMatrix<Complex> Matrix, Complex[] b, Complex[] startX)
        {
            double PieceOfScalarMultiply(Complex[] complexes, int i)
            {
                return complexes[i].Real * complexes[i].Real + complexes[i].Imaginary * complexes[i].Imaginary;
            }

            //set_mas(ggl, ggu, ig1, jg1, di1, n1, pr1, resh);
            var matrix = Factorization.Factorize(FactorizationType, Matrix); //  lu();
            var rightSide = new Complex[b.Length];
            var p = new Complex[b.Length];
            var z = new Complex[b.Length];
            var r = new Complex[b.Length];
            var x = new Complex[b.Length];
            var res = new Complex[b.Length];
            var normR = 0.0;
            var normF = b.Norma();
            Complex alpha, beta, norm_p;
            b.ToList().CopyTo(rightSide);
            startX.ToList().CopyTo(x);

            Matrix.Multiply(x, r,1); //  mult(x, r);
            for (var i = 0; i < rightSide.Length; i++)
            {
                r[i] = rightSide[i] - r[i];
            }

            matrix.Lx(r); //  Lx(r);
            for (var i = 0; i < r.Length; i++)
            {
                z[i] = r[i];
                normR += PieceOfScalarMultiply(r, i);
            }

            matrix.Ux(z); // Ux(z);
            Matrix.Multiply(z, p,1); //mult(z, p);
            matrix.Lx(p);

            normR = Math.Sqrt(normR) / normF;
            norm_p = p.ScalarMultiply(p);
            for (var k = 1; k < MaxIteration && normR > Eps; k++)
            {
                alpha = p.ScalarMultiply(r) / norm_p;
                normR = 0.0;
                for (var i = 0; i < b.Length; i++)
                {
                    x[i] += alpha * z[i];
                    r[i] -= alpha * p[i];
                    rightSide[i] = r[i];
                    normR += PieceOfScalarMultiply(r,i);
                }
                matrix.Ux(rightSide); //       Ux(pr);
                Matrix.Multiply(rightSide, res,1); //       mult(pr, res);
                matrix.Lx(res); //       Lx(res);
                beta = -p.ScalarMultiply(res) / norm_p;
                norm_p = 0.0;
                for (var i = 0; i < b.Length; i++)
                {
                    z[i] = rightSide[i] + beta * z[i];
                    p[i] = res[i] + beta * p[i];
                    norm_p += PieceOfScalarMultiply(p, i);
                }

                normR = Math.Sqrt(normR) / normF;
                //       output(f, k, norm_r);
            }

            return x;
            //  fclose(f);

       
        }
    }
}