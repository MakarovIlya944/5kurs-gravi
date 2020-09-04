using System;
using System.Numerics;
using Quasar.Native;

namespace MKE.Utils {
    public class LinearAlgebra : ILinearAlgebra<double>, ILinearAlgebra<Complex>
    {
        public static ILinearAlgebra<T> Get<T>() => (ILinearAlgebra<T>)new LinearAlgebra();

        double ILinearAlgebra<double>.Cast(double a) => a;
        Complex ILinearAlgebra<Complex>.Cast(double a) => a;

        double ILinearAlgebra<double>.Cast(Complex a) => throw new InvalidCastException();
        Complex ILinearAlgebra<Complex>.Cast(Complex a) => a;

        void ILinearAlgebra<double>.Add(Span<double> a, double alpha, ReadOnlySpan<double> b)
            => BLAS.axpy(a.Length, alpha, b, a);
        void ILinearAlgebra<Complex>.Add(Span<Complex> a, Complex alpha, ReadOnlySpan<Complex> b)
            => BLAS.axpy(a.Length, alpha, b, a);

        void ILinearAlgebra<double>.Sub(Span<double> a, double alpha, ReadOnlySpan<double> b)
            => BLAS.axpy(a.Length, -alpha, b, a);
        void ILinearAlgebra<Complex>.Sub(Span<Complex> a, Complex alpha, ReadOnlySpan<Complex> b)
            => BLAS.axpy(a.Length, -alpha, b, a);

        double ILinearAlgebra<double>.Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
            => BLAS.dot(a.Length, a, b);
        Complex ILinearAlgebra<Complex>.Dot(ReadOnlySpan<Complex> a, ReadOnlySpan<Complex> b)
            => BLAS.dotc(a.Length, a, b);

        double ILinearAlgebra<double>.Norm(ReadOnlySpan<double> a)
            => BLAS.nrm2(a.Length, a);
        double ILinearAlgebra<Complex>.Norm(ReadOnlySpan<Complex> a)
            => BLAS.nrm2(a.Length, a);

        void ILinearAlgebra<double>.Add(ref double a, double b) => a += b;
        void ILinearAlgebra<Complex>.Add(ref Complex a, Complex b) => a += b;
        void ILinearAlgebra<double>.Sub(ref double a, double b) => a -= b;
        void ILinearAlgebra<Complex>.Sub(ref Complex a, Complex b) => a -= b;

        double ILinearAlgebra<double>.Sum(double a, double b) => a + b;
        Complex ILinearAlgebra<Complex>.Sum(Complex a, Complex b) => a + b;

        double ILinearAlgebra<double>.Div(double a, double b) => a / (b + double.Epsilon);
        Complex ILinearAlgebra<Complex>.Div(Complex a, Complex b) => a / (b + double.Epsilon);

        double ILinearAlgebra<double>.Mult(double a, double b) => a * b;
        Complex ILinearAlgebra<Complex>.Mult(Complex a, Complex b) => a * b;

        public static void Add(Span<double> a, double alpha, ReadOnlySpan<double> b)
            => BLAS.axpy(a.Length, alpha, b, a);
        public static void Add(Span<Complex> a, Complex alpha, ReadOnlySpan<Complex> b)
            => BLAS.axpy(a.Length, alpha, b, a);

        public static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
            => BLAS.dot(a.Length, a, b);
        public static Complex Dot(ReadOnlySpan<Complex> a, ReadOnlySpan<Complex> b)
            => BLAS.dotc(a.Length, a, b);

        public static double Norm(ReadOnlySpan<double> a)
            => BLAS.nrm2(a.Length, a);
        public static double Norm(ReadOnlySpan<Complex> a)
            => BLAS.nrm2(a.Length, a);
    }
}