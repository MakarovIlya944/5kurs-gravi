using System;
using System.Collections.Generic;
using System.Linq;

namespace MKE.Extenstion
{
    public static class ListExtenstion
    {
        public static double Multiply(this List<double> a, List<double> b)
        {
            if (a.Count != b.Count) throw new ArgumentException();
            return a.Select((x, i) => x * b[i]).Sum();
        }

        public static double Norma(this List<double> a)
        {
            return Math.Sqrt(a.Multiply(a));
        }

        public static IEnumerable<(T item, int index)> WithIndex<T>(this T[] self) =>
            self?.Select((item, index) => (item, index)) ?? new (T, int)[0];
    }
}