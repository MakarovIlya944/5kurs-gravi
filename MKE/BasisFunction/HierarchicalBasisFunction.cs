using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using MathNet.Numerics.Integration;
using MKE.ElementFragments;
using MKE.Extenstion;
using MKE.Interface;
using MKE.Matrix;

namespace MKE.BasisFunction
{
    public class HierarchicalBasisFunction : IBasisFunction
    {
        //public IEnumerable<Expression<Func<double, double>>> GetBasis1d(int order, double h, double left)
        //{
        //    if (order < 0) throw new Exception("Bad basis order!");

        //    if (order == 0)
        //    {
        //        yield return u => 1;

        //        yield break;
        //    }

        //    yield return u => 1 - (u * h + left);
        //    yield return u => (u * h + left);

        //    if (order == 1)
        //    {
        //        yield break;
        //    }

        //    yield return u => (u * h + left) * ((u * h + left) - 1);

        //    if (order == 2)
        //    {
        //        yield break;
        //    }

        //    yield return u => (u * h + left) * ((u * h + left) - 1) * (2 * (u * h + left) - 1);

        //    if (order == 3)
        //    {
        //        yield break;
        //    }

        //    yield return u => (u * h + left) * ((u * h + left) - 1) * (5 * (u * h + left) * (u * h + left) - 5 * (u * h + left) + 1);

        //    if (order == 4)
        //    {
        //        yield break;
        //    }

        //    yield return u => (u * h + left) * ((u * h + left) - 1) * (2 * (u * h + left) - 1) * (7 * (u * h + left) * (u * h + left) - 7 * (u * h + left) + 1);

        //    if (order == 5)
        //    {
        //        yield break;
        //    }

        //    throw new Exception("Bad basis order!");
        //}
        public IEnumerable<Expression<Func<double, double>>> GetBasis1d(int order, double h, double left)
        {
            if (order < 0) throw new Exception("Bad basis order!");

            if (order == 0)
            {
                yield return u => 1;

                yield break;
            }

            yield return u => 1 - u;
            yield return u => u;

            if (order == 1)
            {
                yield break;
            }

            yield return u => u * (u - 1);

            if (order == 2)
            {
                yield break;
            }

            yield return u => u * (u - 1) * (2 * u - 1);

            if (order == 3)
            {
                yield break;
            }

            yield return u => u * (u - 1) * (5 * u * u - 5 * u + 1);

            if (order == 4)
            {
                yield break;
            }

            yield return u => u * (u - 1) * (2 * u - 1) * (7 * u * u - 7 * u + 1);

            if (order == 5)
            {
                yield break;
            }

            throw new Exception("Bad basis order!");
        }

        public IEnumerable<Expression<Func<double, double, double>>> GetBasis2d(int order, double hu, double hv, double leftU, double leftV)
        {
            return ExpressionExtenstion.Mult(GetBasis1d(order, hu, leftU), GetBasis1d(order, hv, leftV));
        }

        public IEnumerable<Expression<Func<double, double, double, double>>> GetBasis3d(int order, double hu, double hv, double hw, double leftU, double leftV, double leftW)
        {
            return ExpressionExtenstion.Mult(GetBasis1d(order, hu, leftU), GetBasis1d(order, hv, leftV), GetBasis1d(order, hw, leftW));
        }
        public TemplateElementInformation Get2DFragments(int order)
        {
            var points = new List<(int, int)>();

                for (int j = 0; j < order + 1; j++)
                {
                    for (int k = 0; k < order + 1; k++)
                    {
                        points.Add((j, k));
                    }
                }

            var vertex = new List<(int, Vertex)>();

            foreach (var (el, i) in points.WithIndex())
            {
                if (el.Item1 < 2 && el.Item2 < 2)
                    vertex.Add((i, new Vertex(el.LikeBinaryToInt(), order)));
            }

            var edge = new List<(int, Edge)>();

            foreach (var ((x, y), i) in points.WithIndex())
            {
              
                if ((x < 2  && y >= 2))
                    edge.Add((i, new Edge((x, 0).LikeBinaryToInt(), (x, 1).LikeBinaryToInt(), order)));

                if (( y < 2 && x >= 2))
                    edge.Add((i, new Edge((0, y).LikeBinaryToInt(), (1, y).LikeBinaryToInt(), order)));
            }

            var surface = new List<(int, SurfaceSquare)>();
            var inner = new List<int>();

            foreach (var (el, i) in points.WithIndex())
            {
                if (el.Item1 >= 2 && el.Item2 >= 2)
                    inner.Add(i);
            }

            return new TemplateElementInformation(vertex.ToDictionary(x => x.Item1, y => y.Item2), edge.ToDictionary(x => x.Item1, y => y.Item2),
                                                  surface.ToDictionary(x => x.Item1, y => (ISurface)y.Item2), inner.ToArray());
        }

        public TemplateElementInformation Get3DFragments(int order)
        {
            var points = new List<(int, int, int)>();

            for (int i = 0; i < order + 1; i++)
            {
                for (int j = 0; j < order + 1; j++)
                {
                    for (int k = 0; k < order + 1; k++)
                    {
                        points.Add((i, j, k));
                    }
                }
            }

            var vertex = new List<(int, Vertex)>();

            foreach (var (el, i) in points.WithIndex())
            {
                if (el.Item1 < 2 && el.Item2 < 2 && el.Item3 < 2)
                    vertex.Add((i, new Vertex(el.LikeBinaryToInt(), order)));
            }

            var edge = new List<(int, Edge)>();

            foreach (var ((x, y, z), i) in points.WithIndex())
            {
                if (x < 2 && y < 2 && z >= 2)
                    edge.Add((i, new Edge((x, y, 0).LikeBinaryToInt(), (x, y, 1).LikeBinaryToInt(), order)));

                if ((x < 2 && z < 2 && y >= 2))
                    edge.Add((i, new Edge((x, 0, z).LikeBinaryToInt(), (x, 1, z).LikeBinaryToInt(), order)));

                if ((z < 2 && y < 2 && x >= 2))
                    edge.Add((i, new Edge((0, y, z).LikeBinaryToInt(), (1, y, z).LikeBinaryToInt(), order)));
            }

            var surface = new List<(int, SurfaceSquare)>();

            foreach (var ((x, y, z), i) in points.WithIndex())
            {
                if ((x >= 2 && y >= 2 && z < 2))
                    surface.Add((i, new SurfaceSquare((0, 0, z).LikeBinaryToInt(), (0, 1, z).LikeBinaryToInt(), (1, 0, z).LikeBinaryToInt(), (1, 1, z).LikeBinaryToInt(), order)));

                if ((x >= 2 && z >= 2 && y < 2))
                    surface.Add((i, new SurfaceSquare((0, y, 0).LikeBinaryToInt(), (0, y, 1).LikeBinaryToInt(), (1, y, 0).LikeBinaryToInt(), (1, y, 1).LikeBinaryToInt(), order)));

                if ((y >= 2 && z >= 2 && x < 2))
                    surface.Add((i, new SurfaceSquare((x, 0, 0).LikeBinaryToInt(), (x, 0, 1).LikeBinaryToInt(), (x, 1, 0).LikeBinaryToInt(), (x, 1, 1).LikeBinaryToInt(), order)));
            }

            var inner = new List<int>();

            foreach (var (el, i) in points.WithIndex())
            {
                if (el.Item1 >= 2 && el.Item2 >= 2 && el.Item3 >= 2)
                    inner.Add(i);
            }

            return new TemplateElementInformation(vertex.ToDictionary(x => x.Item1, y => y.Item2), edge.ToDictionary(x => x.Item1, y => y.Item2),
                                                  surface.ToDictionary(x => x.Item1, y => (ISurface)y.Item2), inner.ToArray());
        }

        public IEnumerable<Expression<Func<double, double>>> GetDeriveBasis1d(int order, double h, double left)
        {
            if (order < 0) throw new Exception("Bad basis order!");

            if (order == 0)
            {
                yield return u => 0;

                yield break;
            }

            yield return u => -1;
            yield return u => 1;

            if (order == 1)
            {
                yield break;
            }

            yield return u => 2 * u - 1;

            if (order == 2)
            {
                yield break;
            }

            yield return u => 6 * u * (u - 1) + 1;

            if (order == 3)
            {
                yield break;
            }

            yield return u => (2 * u - 1) * (10 * (u - 10) * u + 1);

            if (order == 4)
            {
                yield break;
            }

            yield return u => 1 + 10 * (-1 + u) * u * (2 + 7 * (u - 1) * u);

            if (order == 5)
            {
                yield break;
            }

            throw new Exception("Bad basis order!");
        }
        //public IEnumerable<Expression<Func<double, double>>> GetDeriveBasis1d(int order, double h, double left)
        //{
        //    if (order < 0) throw new Exception("Bad basis order!");

        //    if (order == 0)
        //    {
        //        yield return u => 0;

        //        yield break;
        //    }

        //    yield return u => -1;
        //    yield return u => 1;

        //    if (order == 1)
        //    {
        //        yield break;
        //    }

        //    yield return u => 2 * (u * h + left) - 1;

        //    if (order == 2)
        //    {
        //        yield break;
        //    }

        //    yield return u => 6 * (u * h + left) * ((u * h + left) - 1) + 1;

        //    if (order == 3)
        //    {
        //        yield break;
        //    }

        //    yield return u => (2 * (u * h + left) - 1) * (10 * ((u * h + left) - 10) * (u * h + left) + 1);

        //    if (order == 4)
        //    {
        //        yield break;
        //    }

        //    yield return u => 1 + 10 * (-1 + (u * h + left)) * (u * h + left) * (2 + 7 * ((u * h + left) - 1) * (u * h + left));

        //    if (order == 5)
        //    {
        //        yield break;
        //    }

        //    throw new Exception("Bad basis order!");
        //}

        public IEnumerable<Expression<Func<double, double, double>>> GetDeriveBasis2dU(int order, double hu, double hv, double leftU, double leftV)
        {
            return ExpressionExtenstion.Mult(GetDeriveBasis1d(order, hu, leftU), GetBasis1d(order, hv, leftV));
        }

        public IEnumerable<Expression<Func<double, double, double>>> GetDeriveBasis2dV(int order, double hu, double hv, double leftU, double leftV)
        {
            return ExpressionExtenstion.Mult(GetBasis1d(order, hu, leftU), GetDeriveBasis1d(order, hv, leftV));
        }

        public IEnumerable<Expression<Func<double, double, double, double>>> GetDeriveBasis3dU(int order, double hu, double hv, double hw, double leftU, double leftV, double leftW)
        {
            return ExpressionExtenstion.Mult(GetDeriveBasis1d(order, hu, leftU), GetBasis1d(order, hv, leftV), GetBasis1d(order, hw, leftW));
        }

        public IEnumerable<Expression<Func<double, double, double, double>>> GetDeriveBasis3dV(int order, double hu, double hv, double hw, double leftU, double leftV, double leftW)
        {
            return ExpressionExtenstion.Mult(GetBasis1d(order, hu, leftU), GetDeriveBasis1d(order, hv, leftV), GetBasis1d(order, hw, leftW));
        }

        public IEnumerable<Expression<Func<double, double, double, double>>> GetDeriveBasis3dW(int order, double hu, double hv, double hw, double leftU, double leftV, double leftW)
        {
            return ExpressionExtenstion.Mult(GetBasis1d(order, hu, leftU), GetBasis1d(order, hv, leftV), GetDeriveBasis1d(order, hw, leftW));
        }

        public (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Expression<Func<double, double>> lambda, Expression<Func<double, double>> gamma, int order, double hu, double leftU)
        {
            var basis = GetBasis1d(order, hu, leftU).ToArray();
            var deriveBasis = GetDeriveBasis1d(order, hu, leftU).ToArray();
            var g = new double[basis.Length, basis.Length];
            var m = new double[basis.Length, basis.Length];

            for (int i = 0; i < basis.Length; i++)
            {
                var basisI = basis[i];
                var basisDeriveFunction = deriveBasis[i];

                for (int j = 0; j < basis.Length; j++)
                {
                    var basisIJ = ExpressionExtenstion.Mult(ExpressionExtenstion.Mult(basisI, basis[j]), gamma);
                    var basisDeriveFunctionIJ = ExpressionExtenstion.Mult(ExpressionExtenstion.Mult(basisDeriveFunction, deriveBasis[j]), lambda);
                    g[i, j] = GaussLegendreRule.Integrate(basisDeriveFunctionIJ.Compile(), leftU, leftU + hu, 5);
                    m[i, j] = GaussLegendreRule.Integrate(basisIJ.Compile(), leftU, leftU + hu, 5);
                }
            }

            return (new ReadonlyStorageMatrix(g, basis.Length, basis.Length), new ReadonlyStorageMatrix(m, basis.Length, basis.Length));
        }

        //public (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Expression<Func<double, double, double>> lambda, Expression<Func<double, double, double>> gamma, int order, double hu,
        //    double hv, double leftU, double leftV)
        //{
        //    var basis = GetBasis2d(order, hu, hv, leftU, leftV).ToArray();
        //    var deriveBasisU = GetDeriveBasis2dU(order, hu, hv, leftU, leftV).ToArray();
        //    var deriveBasisV = GetDeriveBasis2dV(order, hu, hv, leftU, leftV).ToArray();
        //    var g = new double[basis.Length, basis.Length];
        //    var m = new double[basis.Length, basis.Length];

        //    for (int i = 0; i < basis.Length; i++)
        //    {
        //        var basisI = basis[i];
        //        var basisDeriveFunctionU = deriveBasisU[i];
        //        var basisDeriveFunctionV = deriveBasisV[i];

        //        for (int j = 0; j < basis.Length; j++)
        //        {
        //            var basisIJ = ExpressionExtenstion.Mult(ExpressionExtenstion.Mult(basisI, basis[j]), gamma);

        //            var basisDeriveFunctionIJ = ExpressionExtenstion
        //                .Mult(ExpressionExtenstion.Add(ExpressionExtenstion.Mult(basisDeriveFunctionU, deriveBasisU[j]), ExpressionExtenstion.Mult(basisDeriveFunctionV, deriveBasisV[j])), lambda);

        //            g[i, j] = GaussLegendreRule.Integrate((x,y)=>basisDeriveFunctionIJ.Compile()(x,y)*hv*hu, leftU, leftU + hu, leftV, leftV + hv, 9);
        //            m[i, j] = GaussLegendreRule.Integrate((x, y) => basisIJ.Compile()(x, y) * hv * hu, leftU, leftU + hu, leftV, leftV + hv, 9);
        //        }
        //    }

        //    return (new ReadonlyStorageMatrix(g, basis.Length, basis.Length), new ReadonlyStorageMatrix(m, basis.Length, basis.Length));
        //}
        public (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Expression<Func<double, double, double>> lambda, Expression<Func<double, double, double>> gamma, int order, double hu,
            double hv, double leftU, double leftV)
        {
            var basis = GetBasis2d(order, hu, hv, leftU, leftV).ToArray();
            var deriveBasisU = GetDeriveBasis2dU(order, hu, hv, leftU, leftV).ToArray();
            var deriveBasisV = GetDeriveBasis2dV(order, hu, hv, leftU, leftV).ToArray();
            var g = new double[basis.Length, basis.Length];
            var m = new double[basis.Length, basis.Length];

            for (int i = 0; i < basis.Length; i++)
            {
                var basisI = basis[i];
                var basisDeriveFunctionU = deriveBasisU[i];
                var basisDeriveFunctionV = deriveBasisV[i];

                for (int j = 0; j < basis.Length; j++)
                {
                    var basisIJ = ExpressionExtenstion.Mult(ExpressionExtenstion.Mult(basisI, basis[j]), gamma);

                    var basisDeriveFunctionIJ = ExpressionExtenstion
                        .Mult(ExpressionExtenstion.Add(ExpressionExtenstion.Mult(basisDeriveFunctionU, deriveBasisU[j]), ExpressionExtenstion.Mult(basisDeriveFunctionV, deriveBasisV[j])), lambda);

                    g[i, j] = GaussLegendreRule.Integrate((x, y) => basisDeriveFunctionIJ.Compile()(x, y) * hv * hu, 0, 1, 0, 1, 9);
                    m[i, j] = GaussLegendreRule.Integrate((x, y) => basisIJ.Compile()(x, y) * hv * hu, 0, 1, 0, 1, 9);
                }
            }

            return (new ReadonlyStorageMatrix(g, basis.Length, basis.Length), new ReadonlyStorageMatrix(m, basis.Length, basis.Length));
        }
        public (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Expression<Func<double, double, double, double>> lambda, Expression<Func<double, double, double, double>> gamma, int order,
            double hu, double hv, double hw, double leftU, double leftV, double leftW)
        {
            var basis = GetBasis3d(order, hu, hv, hw, leftU, leftV, leftW).ToArray();
            var deriveBasisU = GetDeriveBasis3dU(order, hu, hv, hw, leftU, leftV, leftW).ToArray();
            var deriveBasisV = GetDeriveBasis3dV(order, hu, hv, hw, leftU, leftV, leftW).ToArray();
            var deriveBasisW = GetDeriveBasis3dW(order, hu, hv, hw, leftU, leftV, leftW).ToArray();
            var g = new double[basis.Length, basis.Length];
            var m = new double[basis.Length, basis.Length];

            for (int i = 0; i < basis.Length; i++)
            {
                var basisI = basis[i];
                var basisDeriveFunctionU = deriveBasisU[i];
                var basisDeriveFunctionV = deriveBasisV[i];
                var basisDeriveFunctionW = deriveBasisW[i];

                for (int j = 0; j < basis.Length; j++)
                {
                    var basisIJ = ExpressionExtenstion.Mult(ExpressionExtenstion.Mult(basisI, basis[j]), gamma);

                    var basisDeriveFunctionIJ = ExpressionExtenstion
                        .Mult(ExpressionExtenstion.Add(ExpressionExtenstion.Add(ExpressionExtenstion.Mult(basisDeriveFunctionU, deriveBasisU[j]), ExpressionExtenstion.Mult(basisDeriveFunctionV, deriveBasisV[j])), ExpressionExtenstion.Mult(basisDeriveFunctionW, deriveBasisW[j])),
                              lambda);

                    g[i, j] = GaussLegendreRuleExtenstion.Integrate(basisDeriveFunctionIJ.Compile(), leftU, leftU + hu, leftV, leftV + hv, leftW, leftW + hw, 9);
                    m[i, j] = GaussLegendreRuleExtenstion.Integrate(basisIJ.Compile(), leftU, leftU + hu, leftV, leftV + hv, leftW, leftW + hw, 9);
                }
            }

            return (new ReadonlyStorageMatrix(g, basis.Length, basis.Length), new ReadonlyStorageMatrix(m, basis.Length, basis.Length));
        }
    }
}