﻿using System;
using MathNet.Numerics.Integration;
using MKE.Interface;

namespace MKE.Extenstion
{
    public class GaussLegendreRuleExtenstion : GaussLegendreRule
    {
        public GaussLegendreRuleExtenstion(double intervalBegin, double intervalEnd, int order) : base(intervalBegin, intervalEnd, order)
        {
        }

        public static double Integrate(Func<double, double, double, double> f, double xB, double xE, double yB, double yE, double zB, double zE, int order)
        {
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