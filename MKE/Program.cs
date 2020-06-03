using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using MathNet.Numerics;
using MathNet.Numerics.Integration;
using MKE.BasisFunction;
using MKE.Domain;
using MKE.ElementFragments;
using MKE.Extenstion;
using MKE.Interface;
using MKE.Matrix;
using MKE.Point;
using MKE.Solver;

namespace MKE
{
    class Program
    {
        static void Main(string[] args)
        {
            var portrait = new SparsePortrait();

            var GeometryExperement1 = new GeometryParallelepiped();
            GeometryExperement1.MapXAxisLines.Add(1, new AxisLines(0, 1, 1, 2, 0));
            GeometryExperement1.MapYAxisLines.Add(1, new AxisLines(0, 1, 1, 1, 0));
            GeometryExperement1.MapZAxisLines.Add(1, new AxisLines(0, 1, 1, 1, 0));
            GeometryExperement1.MapDomains.Add(1,
                                               new HierarchicalDomain3D<ParallelepipedElement>()
                                               { DomainIndex = 1,RightFunction = (x,y,z)=>0,GeometryParallelepiped = GeometryExperement1, Order = 1, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
         //   GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { F = (x, y, z) =>x, Surface = Surface.Top, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
          //  GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { F = (x, y, z) => x, Surface = Surface.Bottom, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
         //   GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { F = (x, y, z) => x, Surface = Surface.Back, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
            GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { F = (x, y, z) => x, Surface = Surface.Front, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
         //   GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { F = (x, y, z) => x, Surface = Surface.Left, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
         //   GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { F = (x, y, z) => x, Surface = Surface.Right, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
            GeometryExperement1.GeneratePoints();
            var problem = new BaseProblem {Geometry = GeometryExperement1};
            problem.SolveProblem();
           var s= problem.GetSolution(0.25, 0.25, 0.25);
            //for (int i = 0; i < 100; i++)
            //{
            //    var (matrix1, pr, n, _maxiter, eps) = ReadMatrixFromFile("matrixes/1");
            //    var solvrComplex = new ComplexLos();
            //    solvrComplex.Initialization(_maxiter, eps, FactorizationType.LLt);
            //    var sw = new Stopwatch();
            //    var x1 = new Complex[n];
            //    sw.Restart();
            //    solvrComplex.Solve(matrix1, pr, x1);
            //    Console.WriteLine(sw.ElapsedMilliseconds + " ms");
            //}

            Console.ReadKey();
        }
        protected static (double[],bool[]) CalcDirichletCondition(GeometryParallelepiped geometry)
        {

            var portrait = new SparsePortrait();
            foreach (var elem in geometry.DirichletConditions.SelectMany(x => x.Elements))
            {
                portrait.Add(elem.LocalToGlobalEnumeration.Values, elem.LocalToGlobalEnumeration.Values);
            }

            var matrix = new SparseMatrixReal(portrait.GetMappedLinks());
            var b = new double[matrix.Rows];
            var tempIndices = new int[matrix.Rows];
            foreach (var cond in geometry.DirichletConditions)
            {
                foreach (var elem in cond.Elements)
                {
                    var mass = elem.GetMassMatrix();

                    for (int i = 0; i < mass.RowsCount; i++)
                    {
                        for (int j = 0; j < mass.ColumnsCount; j++)
                            matrix.ThreadSafeAdd(portrait.PremutationRowIndices.ElementAt(elem.LocalToGlobalEnumeration[i]), portrait.PremutationColumnIndices.ElementAt(elem.LocalToGlobalEnumeration[j]), mass[i, j]);

                        var value = elem.Integrate(i, cond.F);
                        b.ThreadSafeAdd(portrait.PremutationRowIndices.ElementAt(elem.LocalToGlobalEnumeration[i]), value);
                        tempIndices[portrait.PremutationRowIndices.ElementAt(elem.LocalToGlobalEnumeration[i])] = elem.LocalToGlobalEnumeration[i];
                    }
                }
            }
            var solver = new CGSolver();
            solver.Initialization(1000, 1e-15, FactorizationType.LUsq);
            var tempRightPart= solver.Solve(matrix, b, new double[matrix.Rows]).ToArray();
            var right = new double[geometry.LastSequenceIndex];
            var booleanMask = new bool[geometry.LastSequenceIndex];
            for (var i = 0; i < tempRightPart.Length; i++)
            {
                booleanMask[tempIndices[i]] = true;
                right[tempIndices[i]] = tempRightPart[i];
            }

            return (right,booleanMask);
        }
        private static (IMatrix<Complex>, Complex[], int, int, double) ReadMatrixFromFile(string path)
        {
            int n, _maxiter;
            double eps;

            using (TextReader reader = File.OpenText($"{path}/kuslau"))
            {
                n = int.Parse(reader.ReadLine()) / 2;
                eps = double.Parse(reader.ReadLine(), NumberStyles.AllowExponent | NumberStyles.Float, CultureInfo.InvariantCulture);
                _maxiter = int.Parse(reader.ReadLine());
            }

            var idi = new int[n + 1];

            using (var fs = new FileStream($"{path}/idi", FileMode.Open, FileAccess.Read))
            {
                using var br = new BinaryReader(fs, new ASCIIEncoding());

                for (var i = 0; i < idi.Length; i++)
                {
                    idi[i] = BitConverter.ToInt32(br.ReadBytes(sizeof(int))) - 1;
                }
            }

            var di = new Complex[n];

            using (var fs = new FileStream($"{path}/di", FileMode.Open, FileAccess.Read))
            {
                using var br = new BinaryReader(fs, new ASCIIEncoding());
                var j = 0;

                for (var i = 0; i < idi[n]; i++, j++)
                {
                    var real = BitConverter.ToDouble(br.ReadBytes(sizeof(double)));

                    if (idi[j + 1] - idi[j] == 2)
                    {
                        di[j] = new Complex(real, BitConverter.ToDouble(br.ReadBytes(sizeof(double))));
                        i++;

                        continue;
                    }

                    di[j] = new Complex(real, 0);
                }
            }

            var ig = new int[n + 1];

            using (var fs = new FileStream($"{path}/ig", FileMode.Open, FileAccess.Read))
            {
                using var br = new BinaryReader(fs, new ASCIIEncoding());

                for (var i = 0; i < ig.Length; i++)
                {
                    ig[i] = BitConverter.ToInt32(br.ReadBytes(sizeof(int))) - 1;
                }
            }

            var jg = new int[ig[n]];

            using (var fs = new FileStream($"{path}/jg", FileMode.Open, FileAccess.Read))
            {
                using var br = new BinaryReader(fs, new ASCIIEncoding());

                for (var i = 0; i < jg.Length; i++)
                {
                    jg[i] = BitConverter.ToInt32(br.ReadBytes(sizeof(int))) - 1;
                }
            }

            var ijg = new int[ig[n] + 1];

            using (var fs = new FileStream($"{path}/ijg", FileMode.Open, FileAccess.Read))
            {
                using var br = new BinaryReader(fs, new ASCIIEncoding());

                for (var i = 0; i < ijg.Length; i++)
                {
                    ijg[i] = BitConverter.ToInt32(br.ReadBytes(sizeof(int))) - 1;
                }
            }

            var gg = new Complex[ig[n]];

            using (var fs = new FileStream($"{path}/gg", FileMode.Open, FileAccess.Read))
            {
                using var br = new BinaryReader(fs, new ASCIIEncoding());
                var j = 0;

                for (var i = 0; i < ijg[n]; i++, j++)
                {
                    var real = BitConverter.ToDouble(br.ReadBytes(sizeof(double)));

                    if (ijg[j + 1] - ijg[j] == 2)
                    {
                        gg[j] = new Complex(real, BitConverter.ToDouble(br.ReadBytes(sizeof(double))));
                        i++;

                        continue;
                    }

                    gg[j] = new Complex(real, 0);
                }
            }

            var pr = new Complex[n];

            using (var fs = new FileStream($"{path}/pr", FileMode.Open, FileAccess.Read))
            {
                using var br = new BinaryReader(fs, new ASCIIEncoding());

                for (var i = 0; i < pr.Length; i++)
                {
                    pr[i] = new Complex(BitConverter.ToDouble(br.ReadBytes(sizeof(double))), BitConverter.ToDouble(br.ReadBytes(sizeof(double))));
                }
            }

            return (new SymSparseMatrixComplex(gg, ig, jg, di), pr, n, _maxiter, eps);
        }
    }
}