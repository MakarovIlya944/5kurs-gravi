using System;
using System.Collections.Generic;
using System.Linq;
using MKE.ElementFragments;
using MKE.Extenstion;
using MKE.Interface;
using MKE.Matrix;
using MKE.Point;

namespace MKE.Element {
    public class ParallelepipedElement : IElement
    {
        public IEnumerable<IPointNumbered> Points => PointsReal.Cast<IPointNumbered>();

        public IEnumerable<NumberedPoint3D> PointsReal { get; private set; }

        public IBasisFunction Basis { get; private set; }

        public int Order { get; private set; }
        public int IntegrateOrder { get; private set; }
        public Dictionary<int, int> LocalToGlobalEnumeration { get; set; } = new Dictionary<int, int>();
        public TemplateElementInformation TemplateElementInformation { get; private set; }
        public IEnumerable<Func<double, double, double, double>> BasisFunctions { get; private set; }
        public Func<double, double, double, double> Lambda { get; set; }
        public Func<double, double, double, double> Gamma { get; set; }

        public bool Precalculated { get; set; } = true;
        public ParallelepipedElement(int order, IBasisFunction basis, IEnumerable<NumberedPoint3D> points, int integrateOrder)
        {
            Order = order;
            IntegrateOrder = integrateOrder;
            Basis = basis;
            PointsReal = points;
            TemplateElementInformation = basis.Get3DFragments(order);
            BasisFunctions = Basis.GetBasis3d(Order);
        }
        public Dictionary<int, bool> SkipPoint { get; set; } = new Dictionary<int, bool>(); //in global numeration

        public bool CheckElement(double x, double y, double z)
        {
            var leftFrontBottomPoint = PointsReal.ElementAt(0);
            var rightBackTopPoint = PointsReal.ElementAt(7);

            return leftFrontBottomPoint.X <= x && leftFrontBottomPoint.Y <= y && leftFrontBottomPoint.Z <= z && rightBackTopPoint.X >= x && rightBackTopPoint.Y >= y && rightBackTopPoint.Z >= z;
        }

        public double CalcOnElement(double[] solution, double x, double y, double z)
        {
            var sum = 0d;
            var leftFrontBottomPoint = PointsReal.ElementAt(0);
            var rightBackTopPoint = PointsReal.ElementAt(7);
            var hx = rightBackTopPoint.X - leftFrontBottomPoint.X;
            var hy = rightBackTopPoint.Y - leftFrontBottomPoint.Y;
            var hz = rightBackTopPoint.Z - leftFrontBottomPoint.Z;
            Func<double, double> xTou = (x1) => (x1 - leftFrontBottomPoint.X) / hx;
            Func<double, double> yTov = (x1) => (x1 - leftFrontBottomPoint.Y) / hy;
            Func<double, double> zTow = (x1) => (x1 - leftFrontBottomPoint.Z) / hz;
            foreach (var (key, value) in LocalToGlobalEnumeration)
            {
                sum += BasisFunctions.ElementAt(key)(xTou(x), yTov(y), zTow(z)) * solution[value];
            }

            return sum;
        }

        public void EvaluateLocal(Action<int, int, double> A)
        {
            var leftFrontBottomPoint = PointsReal.ElementAt(0);
            var rightBackTopPoint = PointsReal.ElementAt(7);
            var hx = rightBackTopPoint.X - leftFrontBottomPoint.X;
            var hy = rightBackTopPoint.Y - leftFrontBottomPoint.Y;
            var hz = rightBackTopPoint.Z - leftFrontBottomPoint.Z;
            Func<double, double> uTox = (x) => (x * hx + leftFrontBottomPoint.X);
            Func<double, double> vToy = (x) => (x * hy + leftFrontBottomPoint.Y);
            Func<double, double> wToz = (x) => (x * hz + leftFrontBottomPoint.Z);
            var centered = (rightBackTopPoint - leftFrontBottomPoint / 2);
            (ReadonlyStorageMatrix G, ReadonlyStorageMatrix M) = !Precalculated ?
                Basis.GetMatrix((x, y, z) => Lambda(uTox(x), vToy(y), wToz(z)), (x, y, z) => Gamma(uTox(x), vToy(y), wToz(z)), Order, hx, hy, hz)
                :
                Basis.GetMatrixPreCalculated3d(Lambda(centered.X, centered.Y, centered.Z), Gamma(centered.X, centered.Y, centered.Z), Order, hx, hy, hz);

            for (int i = 0; i < BasisFunctions.Count(); i++)
            {
                if (SkipPoint[LocalToGlobalEnumeration[i]])
                {
                    // B.ThreadSafeSet(LocalToGlobalEnumeration[i], 0);

                    for (int j = 0; j < BasisFunctions.Count(); j++)
                    {
                        A(LocalToGlobalEnumeration[i], LocalToGlobalEnumeration[j], LocalToGlobalEnumeration[i] == LocalToGlobalEnumeration[j] ? 1 : 0);
                    }

                    continue;
                }

                for (int j = 0; j < BasisFunctions.Count(); j++)
                {
                    if (SkipPoint[LocalToGlobalEnumeration[j]])
                    {
                        A(LocalToGlobalEnumeration[i], LocalToGlobalEnumeration[j], LocalToGlobalEnumeration[i] == LocalToGlobalEnumeration[j] ? 1 : 0);
                        continue;
                    }

                    A(LocalToGlobalEnumeration[i], LocalToGlobalEnumeration[j], G[i, j] + M[i, j]);
                }
            }
        }

        public double Integrate(int i, Func<double, double, double, double> func)
        {
            var leftFrontBottomPoint = PointsReal.ElementAt(0);
            var rightBackTopPoint = PointsReal.ElementAt(7);
            var hx = rightBackTopPoint.X - leftFrontBottomPoint.X;
            var hy = rightBackTopPoint.Y - leftFrontBottomPoint.Y;
            var hz = rightBackTopPoint.Z - leftFrontBottomPoint.Z;
            Func<double, double> uTox = (x) => (x * hx + leftFrontBottomPoint.X);
            Func<double, double> vToy = (x) => (x * hy + leftFrontBottomPoint.Y);
            Func<double, double> wToz = (x) => (x * hz + leftFrontBottomPoint.Z);
            return hx * hy * hz * GaussLegendreRuleExtenstion.Integrate((x, y, z) => func(uTox(x), vToy(y), wToz(z)) * BasisFunctions.ElementAt(i)(x, y, z), 0, 1, 0, 1, 0, 1, IntegrateOrder);
        }

        public ReadonlyStorageMatrix GetMassMatrix()
        {
            throw new NotImplementedException();
        }
    }
}