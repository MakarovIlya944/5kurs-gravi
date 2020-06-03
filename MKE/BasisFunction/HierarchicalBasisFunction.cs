using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using MathNet.Numerics.Integration;
using MKE.ElementFragments;
using MKE.Extenstion;
using MKE.Interface;
using MKE.Matrix;

namespace MKE.BasisFunction
{
    public class HierarchicalBasisFunction : IBasisFunction
    {
        private static Dictionary<int, ImmutableArray<Func<double, double>>> PreCompiledFunction1D = CreatePreCompiledFunction1d();

        private static Dictionary<int, ImmutableArray<Func<double, double, double>>> PreCompiledFunction2D = CreatePreCompiledFunction2d();

        private static Dictionary<int, ImmutableArray<Func<double, double, double, double>>> PreCompiledFunction3D = CreatePreCompiledFunction3d();

        private static Dictionary<int, Dictionary<int, TemplateElementInformation>> TemplateInformationDictionary = CreateTemplateInformationDictionary();

        private static Dictionary<int, ImmutableArray<Func<double, double>>> PreCompiledFunction1DDerive = CreatePreCompiledFunction1dDerive();

        private static Dictionary<int, ImmutableArray<Func<double, double, double>>> PreCompiledFunction2DDeriveU = CreatePreCompiledFunction2dDeriveU();

        private static Dictionary<int, ImmutableArray<Func<double, double, double>>> PreCompiledFunction2DDeriveV = CreatePreCompiledFunction2dDeriveV();

        private static Dictionary<int, ImmutableArray<Func<double, double, double, double>>> PreCompiledFunction3DDeriveU = CreatePreCompiledFunction3dDeriveU();

        private static Dictionary<int, ImmutableArray<Func<double, double, double, double>>> PreCompiledFunction3DDeriveV = CreatePreCompiledFunction3dDeriveV();

        private static Dictionary<int, ImmutableArray<Func<double, double, double, double>>> PreCompiledFunction3DDeriveW = CreatePreCompiledFunction3dDeriveW();
        public static Dictionary<int, Dictionary<int, (ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix)>> PreCalculatedMatrix = CreatePreCalculatedMatrix();
        #region Precompilation
        private static Dictionary<int, Dictionary<int, (ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix)>> CreatePreCalculatedMatrix()
        {
            var dict = new Dictionary<int, Dictionary<int, (ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix)>>();

            var innerDict = new Dictionary<int, (ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix)>();

            for (int j = 0; j <= 5; j++)
            {
                innerDict.Add(j, GetMatrixPreCalculated2d(j));
            }

            dict.Add(2, innerDict);
            innerDict = new Dictionary<int, (ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix)>();

            for (int j = 0; j <= 5; j++)
            {
                innerDict.Add(j, GetMatrixPreCalculated3d(j));
            }

            dict.Add(3, innerDict);
            innerDict = new Dictionary<int, (ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix)>();

            for (int j = 0; j <= 5; j++)
            {
                innerDict.Add(j, GetMatrixPreCalculated1d(j));
            }

            dict.Add(1, innerDict);

            return dict;
        }
        private static Dictionary<int, ImmutableArray<Func<double, double>>> CreatePreCompiledFunction1dDerive()
        {
            var innerDict = new Dictionary<int, ImmutableArray<Func<double, double>>>();

            for (int i = 0; i <= 5; i++)
            {
                innerDict.Add(i, GetDeriveBasis1dExpression(i).Select(x => x.Compile()).ToImmutableArray());
            }

            return innerDict;
        }

        private static Dictionary<int, ImmutableArray<Func<double, double, double>>> CreatePreCompiledFunction2dDeriveU()
        {
            var innerDict = new Dictionary<int, ImmutableArray<Func<double, double, double>>>();

            for (int i = 0; i <= 5; i++)
            {
                innerDict.Add(i, GetDeriveBasis2dUExpression(i).Select(x => x.Compile()).ToImmutableArray());
            }

            return innerDict;
        }

        private static Dictionary<int, ImmutableArray<Func<double, double, double>>> CreatePreCompiledFunction2dDeriveV()
        {
            var innerDict = new Dictionary<int, ImmutableArray<Func<double, double, double>>>();

            for (int i = 0; i <= 5; i++)
            {
                innerDict.Add(i, GetDeriveBasis2dVExpression(i).Select(x => x.Compile()).ToImmutableArray());
            }

            return innerDict;
        }

        private static Dictionary<int, ImmutableArray<Func<double, double, double, double>>> CreatePreCompiledFunction3dDeriveU()
        {
            var innerDict = new Dictionary<int, ImmutableArray<Func<double, double, double, double>>>();

            for (int i = 0; i <= 5; i++)
            {
                innerDict.Add(i, GetDeriveBasis3dUExpression(i).Select(x => x.Compile()).ToImmutableArray());
            }

            return innerDict;
        }

        private static Dictionary<int, ImmutableArray<Func<double, double, double, double>>> CreatePreCompiledFunction3dDeriveV()
        {
            var innerDict = new Dictionary<int, ImmutableArray<Func<double, double, double, double>>>();

            for (int i = 0; i <= 5; i++)
            {
                innerDict.Add(i, GetDeriveBasis3dVExpression(i).Select(x => x.Compile()).ToImmutableArray());
            }

            return innerDict;
        }

        private static Dictionary<int, ImmutableArray<Func<double, double, double, double>>> CreatePreCompiledFunction3dDeriveW()
        {
            var innerDict = new Dictionary<int, ImmutableArray<Func<double, double, double, double>>>();

            for (int i = 0; i <= 5; i++)
            {
                innerDict.Add(i, GetDeriveBasis3dWExpression(i).Select(x => x.Compile()).ToImmutableArray());
            }

            return innerDict;
        }

        private static Dictionary<int, Dictionary<int, TemplateElementInformation>> CreateTemplateInformationDictionary()
        {
            var dict = new Dictionary<int, Dictionary<int, TemplateElementInformation>>();

            var innerDict = new Dictionary<int, TemplateElementInformation>();

            for (int j = 0; j <= 5; j++)
            {
                innerDict.Add(j, Get2DFragmentsInner(j));
            }

            dict.Add(2, innerDict);
            innerDict = new Dictionary<int, TemplateElementInformation>();

            for (int j = 0; j <= 5; j++)
            {
                innerDict.Add(j, Get3DFragmentsInner(j));
            }

            dict.Add(3, innerDict);
            innerDict = new Dictionary<int, TemplateElementInformation>();

            for (int j = 0; j <= 5; j++)
            {
                innerDict.Add(j, Get1DFragmentsInner(j));
            }

            dict.Add(1, innerDict);

            return dict;
        }

        private static Dictionary<int, ImmutableArray<Func<double, double>>> CreatePreCompiledFunction1d()
        {
            var innerDict = new Dictionary<int, ImmutableArray<Func<double, double>>>();

            for (int i = 0; i <= 5; i++)
            {
                innerDict.Add(i, GetBasis1dExpression(i).Select(x => x.Compile()).ToImmutableArray());
            }

            return innerDict;
        }

        private static Dictionary<int, ImmutableArray<Func<double, double, double>>> CreatePreCompiledFunction2d()
        {
            var innerDict = new Dictionary<int, ImmutableArray<Func<double, double, double>>>();

            for (int i = 0; i <= 5; i++)
            {
                innerDict.Add(i, GetBasis2dExpression(i).Select(x => x.Compile()).ToImmutableArray());
            }

            return innerDict;
        }

        private static Dictionary<int, ImmutableArray<Func<double, double, double, double>>> CreatePreCompiledFunction3d()
        {
            var innerDict = new Dictionary<int, ImmutableArray<Func<double, double, double, double>>>();

            for (int i = 0; i <= 5; i++)
            {
                innerDict.Add(i, GetBasis3dExpression(i).Select(x => x.Compile()).ToImmutableArray());
            }

            return innerDict;
        }

        #endregion

        #region Public Interface

        public ImmutableArray<Func<double, double>> GetBasis1d(int order) => PreCompiledFunction1D[order];

        public ImmutableArray<Func<double, double, double>> GetBasis2d(int order) => PreCompiledFunction2D[order];

        public ImmutableArray<Func<double, double, double, double>> GetBasis3d(int order) => PreCompiledFunction3D[order];

        public ImmutableArray<Func<double, double>> GetDeriveBasis1d(int order) => PreCompiledFunction1DDerive[order];

        public ImmutableArray<Func<double, double, double>> GetDeriveBasis2dU(int order) => PreCompiledFunction2DDeriveU[order];

        public ImmutableArray<Func<double, double, double>> GetDeriveBasis2dV(int order) => PreCompiledFunction2DDeriveV[order];

        public ImmutableArray<Func<double, double, double, double>> GetDeriveBasis3dU(int order) => PreCompiledFunction3DDeriveU[order];

        public ImmutableArray<Func<double, double, double, double>> GetDeriveBasis3dV(int order) => PreCompiledFunction3DDeriveV[order];

        public ImmutableArray<Func<double, double, double, double>> GetDeriveBasis3dW(int order) => PreCompiledFunction3DDeriveW[order];

        public TemplateElementInformation Get1DFragments(int order) => TemplateInformationDictionary[1][order];

        public TemplateElementInformation Get2DFragments(int order) => TemplateInformationDictionary[2][order];

        public TemplateElementInformation Get3DFragments(int order) => TemplateInformationDictionary[3][order];

        public (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Func<double, double> lambda, Func<double, double> gamma, int order, double hu)
        {
            var basis = GetBasis1d(order);
            var deriveBasis = GetDeriveBasis1d(order);
            var g = new double[basis.Length, basis.Length];
            var m = new double[basis.Length, basis.Length];

            for (int i = 0; i < basis.Length; i++)
            {
                var basisI = basis[i];
                var basisDeriveFunction = deriveBasis[i];

                for (int j = 0; j < basis.Length; j++)
                {
                    var basisJ = basis[j];
                    var basisDeriveFunctionJ = deriveBasis[j];
                    g[i, j] = hu * GaussLegendreRule.Integrate((x) => lambda(x) * basisDeriveFunction(x) * basisDeriveFunctionJ(x), 0, 1, 5);
                    m[i, j] = hu * GaussLegendreRule.Integrate((x) => gamma(x) * basisI(x) * basisJ(x), 0, 1, 5);
                }
            }

            return (new ReadonlyStorageMatrix(g, basis.Length, basis.Length), new ReadonlyStorageMatrix(m, basis.Length, basis.Length));
        }

        public (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Func<double, double, double> lambda, Func<double, double, double> gamma, int order, double hu,
            double hv)
        {
            var basis = GetBasis2d(order);
            var deriveBasisU = GetDeriveBasis2dU(order);
            var deriveBasisV = GetDeriveBasis2dV(order);
            var g = new double[basis.Length, basis.Length];
            var m = new double[basis.Length, basis.Length];

            for (int i = 0; i < basis.Length; i++)
            {
                var basisI = basis[i];
                var basisDeriveFunctionU = deriveBasisU[i];
                var basisDeriveFunctionV = deriveBasisV[i];

                for (int j = 0; j < basis.Length; j++)
                {
                    var basisJ = basis[j];
                    var basisDeriveFunctionUJ = deriveBasisU[j];
                    var basisDeriveFunctionVJ = deriveBasisV[j];

                    g[i, j] = hv * hu *
                              GaussLegendreRule
                                  .Integrate((x, y) => (basisDeriveFunctionU(x, y) * basisDeriveFunctionUJ(x, y) + basisDeriveFunctionV(x, y) * basisDeriveFunctionVJ(x, y)) * lambda(x, y), 0, 1, 0, 1,
                                             6);

                    m[i, j] = hv * hu * GaussLegendreRule.Integrate((x, y) => basisI(x, y) * basisJ(x, y) * gamma(x, y), 0, 1, 0, 1, 6);
                }
            }

            return (new ReadonlyStorageMatrix(g, basis.Length, basis.Length), new ReadonlyStorageMatrix(m, basis.Length, basis.Length));
        }

        public (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Func<double, double, double, double> lambda, Func<double, double, double, double> gamma, int order,
            double hu, double hv, double hw)
        {
            var basis = GetBasis3d(order);
            var deriveBasisU = GetDeriveBasis3dU(order);
            var deriveBasisV = GetDeriveBasis3dV(order);
            var deriveBasisW = GetDeriveBasis3dW(order);
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
                    var basisJ = basis[j];
                    var basisDeriveFunctionUJ = deriveBasisU[j];
                    var basisDeriveFunctionVJ = deriveBasisV[j];
                    var basisDeriveFunctionWJ = deriveBasisW[j];

                    g[i, j] = /*(1/hu) *(1/hv) * (1/hw) **/
                        GaussLegendreRuleExtenstion
                            .Integrate((x, y, z) => hu * hv * hw * ((basisDeriveFunctionU(x, y, z) * basisDeriveFunctionUJ(x, y, z) / hu / hu) + (basisDeriveFunctionV(x, y, z) * basisDeriveFunctionVJ(x, y, z) / hv / hv) + (basisDeriveFunctionW(x, y, z) * basisDeriveFunctionWJ(x, y, z) / hw / hw)) * lambda(x, y, z),
                                       0, 1, 0, 1, 0, 1, 9);

                    m[i, j] = hu * hv * hw * GaussLegendreRuleExtenstion.Integrate((x, y, z) => basisI(x, y, z) * basisJ(x, y, z) * gamma(x, y, z), 0, 1, 0, 1, 0, 1, 9);
                }
            }

            return (new ReadonlyStorageMatrix(g, basis.Length, basis.Length), new ReadonlyStorageMatrix(m, basis.Length, basis.Length));
        }

        public (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrixPreCalculated3d(double lambda, double gamma, int order,
            double hu, double hv, double hw)
        {
            var (g1, g2, g3, m1) = PreCalculatedMatrix[3][order];
            var g = new double[g1.RowsCount, g1.ColumnsCount];

            var m = new double[g1.RowsCount, g1.ColumnsCount];

            for (int i = 0; i < g1.RowsCount; i++)
            {

                for (int j = 0; j < g1.ColumnsCount; j++)
                {

                    g[i, j] = ((g1[i, j] * hv * hw / hu) + (g2[i, j] * hu * hw / hv) + (g3[i, j] * hv * hu / hw)) * lambda;


                    m[i, j] = gamma * hu * hv * hw * m1[i, j];
                }
            }

            return (new ReadonlyStorageMatrix(g, g1.RowsCount, g1.ColumnsCount), new ReadonlyStorageMatrix(m, g1.RowsCount, g1.ColumnsCount));
        }
        public (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrixPreCalculated2d(double lambda, double gamma, int order,
            double hu, double hv)
        {
            var (g1, g2, g3, m1) = PreCalculatedMatrix[2][order];
            var g = new double[g1.RowsCount, g1.ColumnsCount];

            var m = new double[g1.RowsCount, g1.ColumnsCount];

            for (int i = 0; i < g1.RowsCount; i++)
            {

                for (int j = 0; j < g1.ColumnsCount; j++)
                {

                    g[i, j] = ((g1[i, j] * hv / hu) + (g2[i, j] * hu / hv)) * lambda;


                    m[i, j] = gamma * hu * hv * m1[i, j];
                }
            }

            return (new ReadonlyStorageMatrix(g, g1.RowsCount, g1.ColumnsCount), new ReadonlyStorageMatrix(m, g1.RowsCount, g1.ColumnsCount));
        }
        public (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrixPreCalculated1d(double lambda, double gamma, int order,
            double hu)
        {
            var (g1, g2, g3, m1) = PreCalculatedMatrix[1][order];
            var g = new double[g1.RowsCount, g1.ColumnsCount];

            var m = new double[g1.RowsCount, g1.ColumnsCount];

            for (int i = 0; i < g1.RowsCount; i++)
            {

                for (int j = 0; j < g1.ColumnsCount; j++)
                {

                    g[i, j] = (g1[i, j] / hu) * lambda;


                    m[i, j] = gamma * hu * m1[i, j];
                }
            }

            return (new ReadonlyStorageMatrix(g, g1.RowsCount, g1.ColumnsCount), new ReadonlyStorageMatrix(m, g1.RowsCount, g1.ColumnsCount));
        }
        #endregion

        #region Static stuff
        private static (ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrixPreCalculated3d(int order)
        {
            var basis = PreCompiledFunction3D[order];
            var deriveBasisU = PreCompiledFunction3DDeriveU[order];
            var deriveBasisV = PreCompiledFunction3DDeriveV[order];
            var deriveBasisW = PreCompiledFunction3DDeriveW[order];
            var g1 = new double[basis.Length, basis.Length];
            var g2 = new double[basis.Length, basis.Length];
            var g3 = new double[basis.Length, basis.Length];

            var m = new double[basis.Length, basis.Length];

            for (int i = 0; i < basis.Length; i++)
            {
                var basisI = basis[i];
                var basisDeriveFunctionU = deriveBasisU[i];
                var basisDeriveFunctionV = deriveBasisV[i];
                var basisDeriveFunctionW = deriveBasisW[i];

                for (int j = 0; j < basis.Length; j++)
                {
                    var basisJ = basis[j];
                    var basisDeriveFunctionUJ = deriveBasisU[j];
                    var basisDeriveFunctionVJ = deriveBasisV[j];
                    var basisDeriveFunctionWJ = deriveBasisW[j];

                    g1[i, j] = GaussLegendreRuleExtenstion
                        .Integrate((x, y, z) => basisDeriveFunctionU(x, y, z) * basisDeriveFunctionUJ(x, y, z), 0, 1, 0, 1, 0, 1, 9);

                    g2[i, j] = GaussLegendreRuleExtenstion
                        .Integrate((x, y, z) => basisDeriveFunctionV(x, y, z) * basisDeriveFunctionVJ(x, y, z), 0, 1, 0, 1, 0, 1, 9);

                    g3[i, j] = GaussLegendreRuleExtenstion
                        .Integrate((x, y, z) => basisDeriveFunctionW(x, y, z) * basisDeriveFunctionWJ(x, y, z), 0, 1, 0, 1, 0, 1, 9);

                    m[i, j] = GaussLegendreRuleExtenstion.Integrate((x, y, z) => basisI(x, y, z) * basisJ(x, y, z), 0, 1, 0, 1, 0, 1, 9);
                }
            }

            return (new ReadonlyStorageMatrix(g1, basis.Length, basis.Length),
                    new ReadonlyStorageMatrix(g2, basis.Length, basis.Length),
                    new ReadonlyStorageMatrix(g3, basis.Length, basis.Length),
                    new ReadonlyStorageMatrix(m, basis.Length, basis.Length));
        }

        private static (ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrixPreCalculated2d(int order)
        {
            var basis = PreCompiledFunction2D[order];
            var deriveBasisU = PreCompiledFunction2DDeriveU[order];
            var deriveBasisV = PreCompiledFunction2DDeriveV[order];
            var g1 = new double[basis.Length, basis.Length];
            var g2 = new double[basis.Length, basis.Length];

            var m = new double[basis.Length, basis.Length];

            for (int i = 0; i < basis.Length; i++)
            {
                var basisI = basis[i];
                var basisDeriveFunctionU = deriveBasisU[i];
                var basisDeriveFunctionV = deriveBasisV[i];

                for (int j = 0; j < basis.Length; j++)
                {
                    var basisJ = basis[j];
                    var basisDeriveFunctionUJ = deriveBasisU[j];
                    var basisDeriveFunctionVJ = deriveBasisV[j];

                    g1[i, j] = GaussLegendreRuleExtenstion
                        .Integrate((x, y) => basisDeriveFunctionU(x, y) * basisDeriveFunctionUJ(x, y), 0, 1, 0, 1, 9);

                    g2[i, j] = GaussLegendreRuleExtenstion
                        .Integrate((x, y) => basisDeriveFunctionV(x, y) * basisDeriveFunctionVJ(x, y), 0, 1, 0, 1, 9);

                    m[i, j] = GaussLegendreRuleExtenstion.Integrate((x, y) => basisI(x, y) * basisJ(x, y), 0, 1, 0, 1, 9);
                }
            }

            return (new ReadonlyStorageMatrix(g1, basis.Length, basis.Length),
                    new ReadonlyStorageMatrix(g2, basis.Length, basis.Length),
                    new ReadonlyStorageMatrix(new double[basis.Length, basis.Length], basis.Length, basis.Length),
                    new ReadonlyStorageMatrix(m, basis.Length, basis.Length));
        }

        private static (ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrixPreCalculated1d(int order)
        {
            var basis = PreCompiledFunction1D[order];
            var deriveBasisU = PreCompiledFunction1DDerive[order];
            var g1 = new double[basis.Length, basis.Length];

            var m = new double[basis.Length, basis.Length];

            for (int i = 0; i < basis.Length; i++)
            {
                var basisI = basis[i];
                var basisDeriveFunctionU = deriveBasisU[i];

                for (int j = 0; j < basis.Length; j++)
                {
                    var basisJ = basis[j];
                    var basisDeriveFunctionUJ = deriveBasisU[j];

                    g1[i, j] = GaussLegendreRuleExtenstion
                        .Integrate((x) => basisDeriveFunctionU(x) * basisDeriveFunctionUJ(x), 0, 1, 9);


                    m[i, j] = GaussLegendreRuleExtenstion.Integrate((x) => basisI(x) * basisJ(x), 0, 1, 9);
                }
            }

            return (new ReadonlyStorageMatrix(g1, basis.Length, basis.Length),
                    new ReadonlyStorageMatrix(new double[basis.Length, basis.Length], basis.Length, basis.Length),
                    new ReadonlyStorageMatrix(new double[basis.Length, basis.Length], basis.Length, basis.Length),
                    new ReadonlyStorageMatrix(m, basis.Length, basis.Length));
        }

        private static IEnumerable<Expression<Func<double, double>>> GetBasis1dExpression(int order)
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

        private static IEnumerable<Expression<Func<double, double, double>>> GetBasis2dExpression(int order)
        {
            return ExpressionExtenstion.Mult(GetBasis1dExpression(order), GetBasis1dExpression(order));
        }

        private static IEnumerable<Expression<Func<double, double, double, double>>> GetBasis3dExpression(int order)
        {
            return ExpressionExtenstion.Mult(GetBasis1dExpression(order), GetBasis1dExpression(order), GetBasis1dExpression(order));
        }

        private static TemplateElementInformation Get1DFragmentsInner(int order)
        {
            var points = new List<int>();

            for (int k = 0; k < order + 1; k++)
            {
                points.Add(k);
            }

            var vertex = new List<(int, Vertex)>();

            foreach (var (el, i) in points.WithIndex())
            {
                if (el < 2)
                    vertex.Add((i, new Vertex(el, order)));
            }

            var edge = new List<(int, Edge)>();
            var surface = new List<(int, SurfaceSquare)>();
            var inner = new List<int>();

            foreach (var (el, i) in points.WithIndex())
            {
                if (el >= 2)
                    inner.Add(i);
            }

            return new TemplateElementInformation(vertex.ToDictionary(x => x.Item1, y => y.Item2), edge.ToDictionary(x => x.Item1, y => y.Item2),
                                                  surface.ToDictionary(x => x.Item1, y => (ISurface)y.Item2), inner.ToArray());
        }

        private static TemplateElementInformation Get2DFragmentsInner(int order)
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
                if ((x < 2 && y >= 2))
                    edge.Add((i, new Edge((x, 0).LikeBinaryToInt(), (x, 1).LikeBinaryToInt(), order)));

                if ((y < 2 && x >= 2))
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

        private static TemplateElementInformation Get3DFragmentsInner(int order)
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

        private static IEnumerable<Expression<Func<double, double>>> GetDeriveBasis1dExpression(int order)
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

        private static IEnumerable<Expression<Func<double, double, double>>> GetDeriveBasis2dUExpression(int order)
        {
            return ExpressionExtenstion.Mult(GetDeriveBasis1dExpression(order), GetBasis1dExpression(order));
        }

        private static IEnumerable<Expression<Func<double, double, double>>> GetDeriveBasis2dVExpression(int order)
        {
            return ExpressionExtenstion.Mult(GetBasis1dExpression(order), GetDeriveBasis1dExpression(order));
        }

        private static IEnumerable<Expression<Func<double, double, double, double>>> GetDeriveBasis3dUExpression(int order)
        {
            return ExpressionExtenstion.Mult(GetDeriveBasis1dExpression(order), GetBasis1dExpression(order), GetBasis1dExpression(order));
        }

        private static IEnumerable<Expression<Func<double, double, double, double>>> GetDeriveBasis3dVExpression(int order)
        {
            return ExpressionExtenstion.Mult(GetBasis1dExpression(order), GetDeriveBasis1dExpression(order), GetBasis1dExpression(order));
        }

        private static IEnumerable<Expression<Func<double, double, double, double>>> GetDeriveBasis3dWExpression(int order)
        {
            return ExpressionExtenstion.Mult(GetBasis1dExpression(order), GetBasis1dExpression(order), GetDeriveBasis1dExpression(order));
        }

        #endregion
    }
}