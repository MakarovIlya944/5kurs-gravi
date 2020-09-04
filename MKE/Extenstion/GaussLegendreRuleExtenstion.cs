using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using MathNet.Numerics.Integration;
using MKE.Interface;

namespace MKE.Extenstion
{
    public class GaussLegendreRuleExtenstion : GaussLegendreRule
    {
        public static Dictionary<int, (double, double, double, double)[]> WeightsPoints = new Dictionary<int, (double, double, double, double)[]>()
        {
            { 9,GetPreCalculatedWeight(9)}
        };
        public GaussLegendreRuleExtenstion(double intervalBegin, double intervalEnd, int order) : base(intervalBegin, intervalEnd, order)
        {
        }

        private static (double, double, double, double)[] GetPreCalculatedWeight(int order)
        {
            var gaussePointsX = new GaussLegendreRule(0, 1, order);
            var gaussePointsY = new GaussLegendreRule(0, 1, order);
            var gaussePointsZ = new GaussLegendreRule(0, 1, order);
            var array = new (double, double, double, double)[gaussePointsZ.Weights.Length * gaussePointsZ.Weights.Length * gaussePointsZ.Weights.Length];
            var xAbscissas = gaussePointsX.Abscissas;
            var yAbscissas = gaussePointsY.Abscissas;
            var zAbscissas = gaussePointsZ.Abscissas;
            var index = 0;
            foreach (var (wx, i) in gaussePointsX.Weights.WithIndex())
            {
                foreach (var (wy, j) in gaussePointsY.Weights.WithIndex())
                {
                    foreach (var (wz, k) in gaussePointsZ.Weights.WithIndex())
                    {
                        array[index] = (xAbscissas[i], yAbscissas[j], zAbscissas[k], wx * wy * wz);
                        index++;
                    }
                }
            }

            return array;
        }
        public static double Integrate(Func<double, double, double, double> f, double xB, double xE, double yB, double yE, double zB, double zE, int order)
        {
            if (WeightsPoints.ContainsKey(order))
            {
                return WeightsPoints[order].Sum(j => f(j.Item1, j.Item2, j.Item3) * j.Item4);
            }
            var gaussePointsX = new GaussLegendreRule(xB, xE, order);
            var gaussePointsY = new GaussLegendreRule(yB, yE, order);
            var gaussePointsZ = new GaussLegendreRule(zB, zE, order);
            var xAbscissas = gaussePointsX.Abscissas;
            var yAbscissas = gaussePointsY.Abscissas;
            var zAbscissas = gaussePointsZ.Abscissas;

            var sum = 0.0;
            foreach (var (wx, i) in gaussePointsX.Weights.WithIndex())
            {
                foreach (var (wy, j) in gaussePointsY.Weights.WithIndex())
                {
                    foreach (var (wz, k) in gaussePointsZ.Weights.WithIndex())
                    {
                        sum += f(xAbscissas[i], yAbscissas[j], zAbscissas[k]) * wx * wy * wz;
                    }
                }
            }

            return sum;
        }
    }
}