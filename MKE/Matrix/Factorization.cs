using System;
using System.Numerics;
using MKE.Interface;

namespace MKE.Matrix
{
    public enum FactorizationType
    {
        LUsq,
        LLt
    };

    public static class Factorization
    {
        public static IMatrix<T> Factorize<T>(FactorizationType factorization, IMatrix<T> original) where T : unmanaged
        {
            if (typeof(T) == typeof(double))
            {
                switch (original)
                {
                    case SymSparseMatrixReal m1:
                        switch (factorization)
                        {
                            case FactorizationType.LUsq:
                                return (IMatrix<T>)m1.LUsqFactorize();
                            case FactorizationType.LLt:
                                return (IMatrix<T>)m1.LLtFactorize();
                            default:
                                throw new ArgumentOutOfRangeException(nameof(factorization), factorization, null);
                        }
                    case SparseMatrixReal m2:
                        switch (factorization)
                        {
                            case FactorizationType.LUsq:
                                return (IMatrix<T>)m2.LUsqFactorize();
                            case FactorizationType.LLt:
                                throw new ArgumentException("NonSym matrix cant be factorized by LLt");
                            default:
                                throw new ArgumentOutOfRangeException(nameof(factorization), factorization, null);
                        }
                    default:
                        throw new ArgumentOutOfRangeException(nameof(original), original, null);
                }
            }
            if (typeof(T) == typeof(Complex))
            {
                switch (original)
                {
                    case SymSparseMatrixComplex m1:
                        switch (factorization)
                        {
                            case FactorizationType.LUsq:
                                return (IMatrix<T>)m1.LUsqFactorize();
                            case FactorizationType.LLt:
                                return (IMatrix<T>)m1.LLtFactorize();
                            default:
                                throw new ArgumentOutOfRangeException(nameof(factorization), factorization, null);
                        }
                    case SparseMatrixComplex m2:
                        switch (factorization)
                        {
                            case FactorizationType.LUsq:
                                return (IMatrix<T>)m2.LUsqFactorize();
                            case FactorizationType.LLt:
                                throw new ArgumentException("NonSym matrix cant be factorized by LLt");
                            default:
                                throw new ArgumentOutOfRangeException(nameof(factorization), factorization, null);
                        }
                    default:
                        throw new ArgumentOutOfRangeException(nameof(original), original, null);
                }
            }
            throw new ArgumentOutOfRangeException(nameof(original), original, null);
        }
    }
}