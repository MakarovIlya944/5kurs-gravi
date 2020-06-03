using System;
using System.Collections.Generic;
using System.Linq;
using MKE.Utils;

namespace MKE.Point
{
    public class ParallelepipedMesh
    {
        // public List<Ax>
        public AxisLines X { get; private set; }

        public AxisLines Y { get; private set; }

        public AxisLines Z { get; private set; }

        public List<NumberedPoint3D> Points { get; private set; }

        public List<List<int>> Elements { get; private set; }

        public ParallelepipedMesh(AxisLines x, AxisLines y, AxisLines z)
        {
            X = x;
            Y = y;
            Z = z;
            Points = GeneratePoints(x, y, z);
            Elements = GenerateElements(x, y, z);
        }
        public List<NumberedPoint3D> GeneratePoints(AxisLines x, AxisLines y, AxisLines z)
        {
            var k = 0;
            var resultList = new List<NumberedPoint3D>();

            foreach (var zAxis in z.Axises)
            {
                foreach (var yAxis in y.Axises)
                {
                    foreach (var xAxis in x.Axises)
                    {
                        resultList.Add(new NumberedPoint3D(k, xAxis, yAxis, zAxis));
                        k++;
                    }
                }
            }

            return resultList;
        }

        public List<List<int>> GenerateElements(AxisLines x, AxisLines y, AxisLines z)
        {
            var resultList = new List<List<int>>();

            for (var i = 0; i < z.Axises.Count - 1; i++)
            {
                for (var j = 0; j < y.Axises.Count - 1; j++)
                {
                    for (var k = 0; k < x.Axises.Count - 1; k++)
                    {
                        resultList.Add(new List<int>()
                        {
                            k, k + 1,
                            x.Axises.Count * (j + 1) + k, x.Axises.Count * (j + 1) + k + 1,
                            x.Axises.Count * y.Axises.Count * (i + 1) + k, x.Axises.Count * y.Axises.Count * (i + 1) + k + 1,
                            x.Axises.Count * y.Axises.Count * (i + 1) + x.Axises.Count * (j + 1) + k, x.Axises.Count * y.Axises.Count * (i + 1) + x.Axises.Count * (j + 1) + k + 1,
                        });

                        k++;
                    }
                }
            }

            return resultList;
        }
    }

    public class AxisLines
    {
        public List<double> Axises { get; private set; }

        public double Left { get; private set; }

        public double Right { get; private set; }

        public int N { get; private set; }

        private double Q { get; set; }

        private int InnerDerive { get; set; }

        public AxisLines(double left, double right, double q, int n, int innerDerive)
        {
            Left = left;
            Right = right;
            InnerDerive = innerDerive;

            if (n == 0)
            {
                throw new ArgumentException("n can`t be zero", nameof(n));
            }

            if (innerDerive < 0)
            {
                throw new ArgumentException("innerDerive can`t be less zero");
            }

            N = DeriveN(n, innerDerive);
            Q = DeriveQ(q, innerDerive);
            Axises = GenerateAxises(left, right, N, Q);
        }

        private double DeriveQ(double q, int innerDerive)
        {
            return innerDerive == 0 ? q : Math.Pow(q, 1.0 / Math.Pow(2, innerDerive));
        }

        private int DeriveN(int n, int innerDerive)
        {
            return innerDerive == 0 ? n : FunctionUtils.FastPower(2, innerDerive) * n;
        }

        private List<double> GenerateAxises(double a, double b, int n, double q)
        {
            var mas = new List<double>();

            if (Math.Abs(q - 1) > 1e-14)
            {
                var h = (1d - q) * (b - a) / (1 - Math.Pow(q, n));
                var s = h;
                var sNext = a;

                for (var i = 0; i < n; i++)
                {
                    mas.Add(sNext);
                    s *= q;
                    sNext += s;
                }
                mas.Add(b);
            }
            else
            {
                var h = (b - a) / n;

                for (var i = 0; i < n; i++)
                {
                    mas.Add(a + h * i);
                }
                mas.Add(b);
            }

            return mas;
        }
    }
}