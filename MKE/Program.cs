using System;
using System.Collections.Generic;
using System.IO;
using MKE.Geometry;
using MKE.Interface;
using MKE.Point;
using MKE.Problem;
using MKE.Problem.InverseProblem;
using Newtonsoft.Json;

namespace MKE
{
    class Program
    {
        static void Main(string[] args)
        {
            for (var j = 2; j < 17; j++)
            {
                List<ValuedPoint3D> points = new List<ValuedPoint3D>();

                using (StreamReader r = new StreamReader("problem.json"))
                {
                    string json = r.ReadToEnd();
                    var geometry = JsonConvert.DeserializeObject<GeometryParallelepiped>(json);
                    geometry.GeneratePoints();
                    var problem = new BaseProblem(geometry);
                    problem.SolveProblem();
                   // Console.Clear();
                    geometry.StepsT = j;

                    var h = 1d / Convert.ToDouble(geometry.StepsT);

                    for (var i = 1; i <= geometry.StepsT; i++)
                    {
                        var x = geometry.FuncTransformX(h * i);
                        var y = geometry.FuncTransformY(h * i);
                        var z = geometry.FuncTransformZ(h * i);

                        if (i == 1)
                        {
                            points.Add(new ValuedPoint3D() {X = x, Y = y, Z = z, Value = problem.GetSolution(x, y, z)*1.05});
                        }
                        else
                        {
                            points.Add(new ValuedPoint3D() {X = x, Y = y, Z = z, Value = problem.GetSolution(x, y, z)});
                        }
                    }
                }

                using (StreamReader r = new StreamReader("inverseProblem.json"))
                {
                    string json = r.ReadToEnd();
                    var inverseConfiguration = JsonConvert.DeserializeObject<InverseConfiguration>(json);
                    inverseConfiguration.ExperimentalPoints = points;

                    using (StreamReader r1 = new StreamReader(inverseConfiguration.ProblemPath))
                    {
                        json = r1.ReadToEnd();
                        var geometry = JsonConvert.DeserializeObject<GeometryParallelepiped>(json);

                        geometry.GeneratePoints();
                        var problem = new BaseProblem(geometry);
                        var gaussNewtonProblem = new GaussNewtonInverseProblem(problem, inverseConfiguration);
                        gaussNewtonProblem.Solve();
                        //  Console.WriteLine(JsonConvert.SerializeObject(gaussNewtonProblem.Solve()));
                    }
                }
            }

            /*using (StreamReader r = new StreamReader("problem.json"))
            {
                string json = r.ReadToEnd();
                var GeometryExperement1 = JsonConvert.DeserializeObject<GeometryParallelepiped>(json);

                Console.Write(JsonConvert.SerializeObject(GeometryExperement1, Formatting.Indented,
                                                          new JsonSerializerSettings()
                                                          {
                                                              ReferenceLoopHandling = ReferenceLoopHandling.Ignore
                                                          }));

                GeometryExperement1.GeneratePoints();
                var problem = new BaseProblem { Geometry = GeometryExperement1 };
                problem.PrepareProblem();
               problem.SolveProblem();
                Console.Clear();
                var h = 1d / Convert.ToDouble(GeometryExperement1.StepsT);
                var list = new List<ValuedPoint3D>();
                for (var i = 1; i <= GeometryExperement1.StepsT; i++)
                {
                    var x = GeometryExperement1.FuncTransformX(h * i);
                    var y = GeometryExperement1.FuncTransformY(h * i);
                    var z = GeometryExperement1.FuncTransformZ(h * i);
                    list.Add(new ValuedPoint3D() { X = x, Y = y, Z = z, Value = problem.GetSolution(x, y, z) });
                }
                Console.WriteLine(JsonConvert.SerializeObject(list));
            }
            */
            Console.ReadKey();
        }
    }
}