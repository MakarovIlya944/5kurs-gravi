using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading;

namespace MKE.Extenstion
{
    public static class ArrayExtenstion
    {
        public static double ScalarMultiply(this double[] a, double[] b)
        {
            if (a.Length != b.Length) throw new ArgumentException();
            return a.Select((x, i) => x * b[i]).Sum();
        }

        public static Complex Multiply(this Complex[] a, Complex[] b)
        {
            if (a.Length != b.Length) throw new ArgumentException();
            return a.Select((x, i) => x * b[i]).Sum();
        }

        public static Complex Sum(this IEnumerable<Complex> a)
        {
            return a.Aggregate(new Complex(), (current, value) => current + value);
        }

        public static double Norma(this Complex[] a)
        {
            return Math.Sqrt(a.ScalarMultiply(a));
        }

        public static double Norma(this Complex[] a, Complex[] b)
        {
            return Math.Sqrt(a.ScalarMultiply(b));
        }

        public static double Norma(this double[] a)
        {
            return Norma(a, a);
        }

        public static double Norma(this double[] a, double[] b)
        {
            return Math.Sqrt(a.ScalarMultiply(b));
        }

        public static Complex BiLinearFormConjugate(this Complex[] a, Complex[] b)
        {
            if (a.Length != b.Length) throw new ArgumentException();
            return a.Select((value, i) => new Complex(value.Real * b[i].Real - value.Imaginary * b[i].Imaginary, value.Real * b[i].Imaginary + value.Imaginary * b[i].Real)).Sum();

        }

        public static Complex BiLinearForm(this Complex[] a, Complex[] b)
        {
            if (a.Length != b.Length) throw new ArgumentException();
            return a.Select((value, i) => new Complex(value.Real * b[i].Real + value.Imaginary * b[i].Imaginary, value.Real * b[i].Imaginary - value.Imaginary * b[i].Real)).Sum();
        }

        public static double ScalarMultiply(this Complex[] a, Complex[] b)
        {
            if (a.Length != b.Length) throw new ArgumentException();
            return a.Select((t, i) => t.Real * b[i].Real + t.Imaginary * b[i].Imaginary).Sum();
        }

        public static IEnumerable<(T item, int index)> WithIndex<T>(this IEnumerable<T> self) =>
            self?.Select((item, index) => (item, index)) ?? new List<(T, int)>();

        public static void ThreadSafeAdd(this double[] array, int index, double value)
        {
            if (double.IsNaN(value))
                throw new ArgumentException("Value is NAN!");
            double initialValue, computedValue;
            do
            {
                initialValue = array[index];
                computedValue = initialValue + value;
            } while (initialValue != Interlocked.CompareExchange(ref array[index], computedValue, initialValue));
        }
        public static void ThreadSafeSet(this double[] array, int index, double value)
        {
            if (double.IsNaN(value))
                throw new ArgumentException("Value is NAN!");
            double initialValue, computedValue;
            do
            {
                initialValue = array[index];
                computedValue = value;
            } while (initialValue != Interlocked.CompareExchange(ref array[index], computedValue, initialValue));
        }
        public static void ThreadSafeAdd(this Complex[] array, int index, Complex value)
        {
            if (Complex.IsNaN(value))
                throw new ArgumentException("Value is NAN!");
            lock ((object)array[index])
            {
                array[index] = array[index] + value;
            }
        }
        public static void ThreadSafeSet(this Complex[] array, int index, Complex value)
        {
            if (Complex.IsNaN(value))
                throw new ArgumentException("Value is NAN!");
            lock ((object)array[index])
            {
                array[index] = value;
            }
        }
    }
}