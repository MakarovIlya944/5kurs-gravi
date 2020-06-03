using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using MathNet.Numerics.Integration;
using MKE.ElementFragments;
using MKE.Extenstion;
using MKE.Matrix;
using MKE.Point;

namespace MKE.Interface
{
    public interface IElement
    {
        IBasisFunction Basis { get; }

        IEnumerable<IPointNumbered> Points { get; }

        Dictionary<int, int> LocalToGlobalEnumeration { get; }

        Dictionary<int, bool> SkipPoint { get; }
        TemplateElementInformation TemplateElementInformation { get; }
        void EvaluateLocal(Action<int, int, double> A);

        double Integrate(int i, Func<double, double, double, double> func);
        ReadonlyStorageMatrix GetMassMatrix();

        bool CheckElement(double x, double y, double z);

        double CalcOnElement(double[] solution, double x, double y, double z);
    }

    public class SquaredElement : IElement
    {
        public IBasisFunction Basis { get; private set; }
        public int Order { get; private set; }
        public int IntegrateOrder { get; private set; }

        public Surface Surface { get; private set; }
        public IEnumerable<IPointNumbered> Points => PointsReal.Cast<IPointNumbered>();
        public IEnumerable<NumberedPoint3D> PointsReal { get; private set; }
        public Dictionary<int, int> LocalToGlobalEnumeration { get; } = new Dictionary<int, int>();
        public Dictionary<int, bool> SkipPoint { get; } = new Dictionary<int, bool>();
        public IEnumerable<Func<double, double, double>> BasisFunctions { get; private set; }

        public TemplateElementInformation TemplateElementInformation { get;private set; }
        public SquaredElement(int order, IBasisFunction basis, IEnumerable<NumberedPoint3D> points, int integrateOrder, Surface surface)
        {
            Order = order;
            IntegrateOrder = integrateOrder;
            Basis = basis;
            PointsReal = points;
            Surface = surface;
            TemplateElementInformation = basis.Get2DFragments(order);
            var leftFrontBottomPoint = PointsReal.ElementAt(0);
            var rightBackTopPoint = PointsReal.ElementAt(3);
            var hx = rightBackTopPoint.X - leftFrontBottomPoint.X;
            var hy = rightBackTopPoint.Y - leftFrontBottomPoint.Y;
            var hz = rightBackTopPoint.Z - leftFrontBottomPoint.Z;
            BasisFunctions = Surface switch
            {
                Surface.Front => Basis.GetBasis2d(Order, hx, hz, leftFrontBottomPoint.X, leftFrontBottomPoint.Z).Select(x => x.Compile()),
                Surface.Back => Basis.GetBasis2d(Order, hx, hz, leftFrontBottomPoint.X, leftFrontBottomPoint.Z).Select(x => x.Compile()),
                Surface.Top => Basis.GetBasis2d(Order, hx, hy, leftFrontBottomPoint.X, leftFrontBottomPoint.Y).Select(x => x.Compile()),
                Surface.Bottom => Basis.GetBasis2d(Order, hx, hy, leftFrontBottomPoint.X, leftFrontBottomPoint.Y).Select(x => x.Compile()),
                Surface.Left => Basis.GetBasis2d(Order, hy, hz, leftFrontBottomPoint.Y, leftFrontBottomPoint.Z).Select(x => x.Compile()),
                Surface.Right => Basis.GetBasis2d(Order, hy, hz, leftFrontBottomPoint.Y, leftFrontBottomPoint.Z).Select(x => x.Compile()),
                _ => throw new ArgumentOutOfRangeException()
            };
        }
        public double Integrate(int i, Func<double, double, double, double> func)
        {
            var leftFrontBottomPoint = PointsReal.ElementAt(0);
            var rightBackTopPoint = PointsReal.ElementAt(3);
            var hx = rightBackTopPoint.X - leftFrontBottomPoint.X;
            var hy = rightBackTopPoint.Y - leftFrontBottomPoint.Y;
            var hz = rightBackTopPoint.Z - leftFrontBottomPoint.Z;
            Func<double, double> uTox = (x) => (x * hx + leftFrontBottomPoint.X);
            Func<double, double> vToy = (x) => (x * hy + leftFrontBottomPoint.Y);
            Func<double, double> wToz = (x) => (x * hz + leftFrontBottomPoint.Z);

           return Surface switch
            {
                Surface.Front => GaussLegendreRule.Integrate((x, y) => func(uTox(x), leftFrontBottomPoint.Y, wToz(x)) * BasisFunctions.ElementAt(i)(x, y) * hx * hz, 0, 1, 0, 1, IntegrateOrder),
                Surface.Back => GaussLegendreRule.Integrate((x, y) => func(uTox(x), rightBackTopPoint.Y, wToz(x)) * BasisFunctions.ElementAt(i)(x, y) * hx * hz, 0, 1, 0, 1, IntegrateOrder),
                Surface.Top => GaussLegendreRule.Integrate((x, y) => func(uTox(x), vToy(y), rightBackTopPoint.Z) * BasisFunctions.ElementAt(i)(x, y) * hx * hy, 0, 1, 0, 1, IntegrateOrder),
                Surface.Bottom => GaussLegendreRule.Integrate((x, y) => func(uTox(x), vToy(y), leftFrontBottomPoint.Z) * BasisFunctions.ElementAt(i)(x, y) * hx * hy, 0, 1, 0, 1, IntegrateOrder),
                Surface.Left => GaussLegendreRule.Integrate((x, y) => func(leftFrontBottomPoint.X, vToy(x), wToz(y)) * BasisFunctions.ElementAt(i)(x, y) * hy * hz, 0, 1, 0, 1, IntegrateOrder),
                Surface.Right => GaussLegendreRule.Integrate((x, y) => func(rightBackTopPoint.X, vToy(x), wToz(y)) * BasisFunctions.ElementAt(i)(x, y) * hy * hz, 0, 1, 0, 1, IntegrateOrder),
                _ => throw new ArgumentOutOfRangeException()
            };
        }
        public ReadonlyStorageMatrix GetMassMatrix()
        {
            var leftFrontBottomPoint = PointsReal.ElementAt(0);
            var rightBackTopPoint = PointsReal.ElementAt(3);
            var hx = rightBackTopPoint.X - leftFrontBottomPoint.X;
            var hy = rightBackTopPoint.Y - leftFrontBottomPoint.Y;
            var hz = rightBackTopPoint.Z - leftFrontBottomPoint.Z;

            return Surface switch
            {
                Surface.Front => Basis.GetMatrix((x, y) => 1, (x, y) => 1, Order, hx, hz, leftFrontBottomPoint.X, leftFrontBottomPoint.Z).Item2,
                Surface.Back => Basis.GetMatrix((x, y) => 1, (x, y) => 1, Order, hx, hz, leftFrontBottomPoint.X, leftFrontBottomPoint.Z).Item2,
                Surface.Top => Basis.GetMatrix((x, y) => 1, (x, y) => 1, Order, hx, hy, leftFrontBottomPoint.X, leftFrontBottomPoint.Y).Item2,
                Surface.Bottom => Basis.GetMatrix((x, y) => 1, (x, y) => 1, Order, hx, hy, leftFrontBottomPoint.X, leftFrontBottomPoint.Y).Item2,
                Surface.Left => Basis.GetMatrix((x, y) => 1, (x, y) => 1, Order, hy, hz, leftFrontBottomPoint.Y, leftFrontBottomPoint.Z).Item2,
                Surface.Right => Basis.GetMatrix((x, y) => 1, (x, y) => 1, Order, hy, hz, leftFrontBottomPoint.Y, leftFrontBottomPoint.Z).Item2,
                _ => throw new ArgumentOutOfRangeException()
            };
        }
        public void EvaluateLocal(Action<int, int, double> A)
        {
            throw new NotImplementedException();
        }
        public bool CheckElement(double x, double y, double z)
        {
            var leftFrontBottomPoint = PointsReal.ElementAt(0);
            var rightBackTopPoint = PointsReal.ElementAt(7);

            return leftFrontBottomPoint.X >= x && leftFrontBottomPoint.Y >= y && leftFrontBottomPoint.Z >= z && rightBackTopPoint.X <= x && rightBackTopPoint.Y <= y && rightBackTopPoint.Z <= z;
        }

        public double CalcOnElement(double[] solution, double x, double y, double z)
        {
            throw new NotImplementedException();
        }
    }
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
        public IEnumerable<Func<double, double, double, double>> DeriveBasisU { get; private set; }
        public IEnumerable<Func<double, double, double, double>> DeriveBasisV { get; private set; }
        public IEnumerable<Func<double, double, double, double>> DeriveBasisW { get; private set; }

        public ParallelepipedElement(int order, IBasisFunction basis, IEnumerable<NumberedPoint3D> points,int integrateOrder)
        {
            Order = order;
            IntegrateOrder = integrateOrder;
            Basis = basis;
            PointsReal = points;
            TemplateElementInformation = basis.Get3DFragments(order);
            var leftFrontBottomPoint = PointsReal.ElementAt(0);
            var rightBackTopPoint = PointsReal.ElementAt(7);
            var hx = rightBackTopPoint.X - leftFrontBottomPoint.X;
            var hy = rightBackTopPoint.Y - leftFrontBottomPoint.Y;
            var hz = rightBackTopPoint.Z - leftFrontBottomPoint.Z;
            BasisFunctions = Basis.GetBasis3d(Order, hx, hy, hz, leftFrontBottomPoint.X, leftFrontBottomPoint.Y, leftFrontBottomPoint.Z).Select(x => x.Compile()).ToArray();
            DeriveBasisU = Basis.GetDeriveBasis3dU(order, hx, hy, hz, leftFrontBottomPoint.X, leftFrontBottomPoint.Y, leftFrontBottomPoint.Z).Select(x => x.Compile()).ToArray();
            DeriveBasisV = Basis.GetDeriveBasis3dV(order, hx, hy, hz, leftFrontBottomPoint.X, leftFrontBottomPoint.Y, leftFrontBottomPoint.Z).Select(x => x.Compile()).ToArray();
            DeriveBasisW = Basis.GetDeriveBasis3dW(order, hx, hy, hz, leftFrontBottomPoint.X, leftFrontBottomPoint.Y, leftFrontBottomPoint.Z).Select(x => x.Compile()).ToArray();
        }
        public Dictionary<int, bool> SkipPoint { get; set; } = new Dictionary<int, bool>(); //in global numeration

        public bool CheckElement(double x, double y,double z)
        {
            var leftFrontBottomPoint = PointsReal.ElementAt(0);
            var rightBackTopPoint = PointsReal.ElementAt(7);

            return leftFrontBottomPoint.X < x && leftFrontBottomPoint.Y < y && leftFrontBottomPoint.Z < z && rightBackTopPoint.X >x && rightBackTopPoint.Y > y && rightBackTopPoint.Z > z;
        }

        public double CalcOnElement(double[] solution, double x, double y, double z)
        {
            var sum = 0d;

            foreach (var (key,value) in LocalToGlobalEnumeration)
            {
                sum += BasisFunctions.ElementAt(key)(x, y, z) * solution[value];
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

            (ReadonlyStorageMatrix G, ReadonlyStorageMatrix M) =
                Basis.GetMatrix((x, y, z) => 1, (x, y, z) => 1, Order, hx, hy, hz, leftFrontBottomPoint.X, leftFrontBottomPoint.Y, leftFrontBottomPoint.Z);

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

                //var integrateFunc = ExpressionExtenstion.Mult(basisFunction[i], rightPart);

                //B.ThreadSafeAdd(LocalToGlobalEnumeration[i],
                //                GaussLegendreRuleExtenstion.Integrate(integrateFunc.Compile(), leftFrontBottomPoint.X, rightBackTopPoint.X, leftFrontBottomPoint.Y, rightBackTopPoint.Y,
                //                                                      leftFrontBottomPoint.Z, rightBackTopPoint.Z, 5));
                var basisI = BasisFunctions.ElementAt(i);
                var basisDeriveFunctionU = DeriveBasisU.ElementAt(i);
                var basisDeriveFunctionV = DeriveBasisV.ElementAt(i);
                var basisDeriveFunctionW = DeriveBasisW.ElementAt(i);
                for (int j = 0; j < BasisFunctions.Count(); j++)
                {
                    if (SkipPoint[LocalToGlobalEnumeration[j]])
                    {
                        A(LocalToGlobalEnumeration[i], LocalToGlobalEnumeration[j], LocalToGlobalEnumeration[i] == LocalToGlobalEnumeration[j] ? 1 : 0);
                        continue;
                    }

                    var basisJ = BasisFunctions.ElementAt(j);
                    var DeriveFunctionU = DeriveBasisU.ElementAt(j);
                    var DeriveFunctionV = DeriveBasisV.ElementAt(j);
                    var DeriveFunctionW = DeriveBasisW.ElementAt(j);
                    var g= GaussLegendreRuleExtenstion.Integrate((x,y,z)=> (basisDeriveFunctionU(x,y,z)* DeriveFunctionU(x,y,z)+ basisDeriveFunctionV(x, y, z) * DeriveFunctionV(x, y, z)+ basisDeriveFunctionW(x, y, z) * DeriveFunctionW(x, y, z)), leftFrontBottomPoint.X, rightBackTopPoint.X, leftFrontBottomPoint.Y, rightBackTopPoint.Y, leftFrontBottomPoint.Z, rightBackTopPoint.Z, 5);
                    var m = GaussLegendreRuleExtenstion.Integrate((x,y,z)=>(basisI(x,y,z)*basisJ(x,y,z)*1), leftFrontBottomPoint.X, rightBackTopPoint.X, leftFrontBottomPoint.Y, rightBackTopPoint.Y, leftFrontBottomPoint.Z, rightBackTopPoint.Z, 5);

                    A(LocalToGlobalEnumeration[i], LocalToGlobalEnumeration[j],g /*+ M[i, j]*/);
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
            return GaussLegendreRuleExtenstion.Integrate((x, y, z) =>func(x, y, z) * BasisFunctions.ElementAt(i)(x, y, z),leftFrontBottomPoint.X, rightBackTopPoint.X, leftFrontBottomPoint.Y, rightBackTopPoint.Y, leftFrontBottomPoint.Z, rightBackTopPoint.Z,IntegrateOrder);
        }

        public ReadonlyStorageMatrix GetMassMatrix()
        {
            throw new NotImplementedException();
        }
    }
}