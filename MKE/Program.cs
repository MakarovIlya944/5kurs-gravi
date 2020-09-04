using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq.Dynamic.Core;
using System.Linq.Expressions;
using System.Numerics;
using System.Text;
using MKE.Domain;
using MKE.Geometry;
using MKE.Interface;
using MKE.Matrix;
using MKE.Point;
using MKE.Problem;
using Newtonsoft.Json;
namespace MKE
{
    class Program
    {
        static void Main(string[] args)
        {
            using (StreamReader r = new StreamReader("problem.json"))
            {
                string json = r.ReadToEnd();
                var GeometryExperement1 = JsonConvert.DeserializeObject<GeometryParallelepiped>(json);
                GeometryExperement1.InitAfterSerialize();

                Console.Write(JsonConvert.SerializeObject(GeometryExperement1, Formatting.Indented,
                                                          new JsonSerializerSettings()
                                                          {
                                                              ReferenceLoopHandling = ReferenceLoopHandling.Ignore
                                                          }));
                GeometryExperement1.GeneratePoints();
                var problem = new BaseProblem { Geometry = GeometryExperement1 };
                problem.SolveProblem();
                Console.Clear();
                var h = 1d / Convert.ToDouble(GeometryExperement1.StepsT);
                for (var i = 1; i <= GeometryExperement1.StepsT; i++)
                {
                    var x = GeometryExperement1.FuncTransformX(h * i);
                    var y = GeometryExperement1.FuncTransformY(h * i);
                    var z = GeometryExperement1.FuncTransformZ(h * i);
                    //Console.WriteLine($"{x}\t{y}\t{z}\t{problem.GetSolution(x, y, z)}");
                    Console.WriteLine($"{problem.GetSolution(x, y, z)}");
                }
            }
            //var GeometryExperement1 = new GeometryParallelepiped();
            //GeometryExperement1.MapXAxisLines.Add(1, new AxisLines(0, 1, 1, 2, 0));
            //GeometryExperement1.MapYAxisLines.Add(1, new AxisLines(0, 1, 1, 2, 0));
            //GeometryExperement1.MapZAxisLines.Add(1, new AxisLines(0, 1, 1, 2, 0));
            //GeometryExperement1.MapXAxisLines.Add(2, new AxisLines(1, 2, 1, 2, 0));

            //GeometryExperement1.MapDomains.Add(1,
            //                                   new HierarchicalDomain3D<ParallelepipedElement>()
            //                                   {
            //                                       DomainIndex = 1,
            //                                       GammaFunction = "1.0",
            //                                       LambdaFunction = "1.0",
            //                                       Function = "-2.0 + X * X",
            //                                       GeometryParallelepiped = GeometryExperement1,
            //                                       Order = 2,
            //                                       XAxisIndex = 1,
            //                                       YAxisIndex = 1,
            //                                       ZAxisIndex = 1
            //                                   });

            //GeometryExperement1.MapDomains.Add(2,
            //                                   new HierarchicalDomain3D<ParallelepipedElement>()
            //                                   {
            //                                       DomainIndex = 2,
            //                                       GammaFunction = "1.0",
            //                                       LambdaFunction = "1.0",
            //                                       Function = "-2.0 + X * X",
            //                                       GeometryParallelepiped = GeometryExperement1,
            //                                       Order = 3,
            //                                       XAxisIndex = 2,
            //                                       YAxisIndex = 1,
            //                                       ZAxisIndex = 1
            //                                   });

            //GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { Function = "X * X", Surface = Surface.Top, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
            //GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { Function = "X * X", Surface = Surface.Bottom, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
            //GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { Function = "X * X", Surface = Surface.Back, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
            //GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { Function = "X * X", Surface = Surface.Front, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
            //GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { Function = "X * X", Surface = Surface.Left, XAxisIndex = 1, YAxisIndex = 1, ZAxisIndex = 1 });
            //GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { Function = "X * X", Surface = Surface.Top, XAxisIndex = 2, YAxisIndex = 1, ZAxisIndex = 1 });
            //GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { Function = "X * X", Surface = Surface.Bottom, XAxisIndex = 2, YAxisIndex = 1, ZAxisIndex = 1 });
            //GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { Function = "X * X", Surface = Surface.Back, XAxisIndex = 2, YAxisIndex = 1, ZAxisIndex = 1 });
            //GeometryExperement1.DirichletConditions.Add(new DirichletCondition() { Function = "X * X", Surface = Surface.Front, XAxisIndex = 2, YAxisIndex = 1, ZAxisIndex = 1 });
            //GeometryExperement1.NeumannConditions.Add(new NeumannCondition() { Function = "2.0 * X", Surface = Surface.Right, XAxisIndex = 2, YAxisIndex = 1, ZAxisIndex = 1 });

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