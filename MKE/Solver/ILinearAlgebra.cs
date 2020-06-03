using System;
using System.Numerics;

namespace MKE.Solver {
    public interface ILinearAlgebra<T>
    {
        T Cast(double a);
        T Cast(Complex a);

        T Sum(T a, T b);
        T Div(T a, T b);
        T Mult(T a, T b);

        void Add(ref T a, T b);
        void Sub(ref T a, T b);

        void Add(Span<T> a, T alpha, ReadOnlySpan<T> b);
        void Sub(Span<T> a, T alpha, ReadOnlySpan<T> b);
        T Dot(ReadOnlySpan<T> a, ReadOnlySpan<T> b);
        double Norm(ReadOnlySpan<T> a);
    }
}