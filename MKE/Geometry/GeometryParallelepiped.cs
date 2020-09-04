using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Dynamic.Core;
using MKE.BasisFunction;
using MKE.Condition;
using MKE.Domain;
using MKE.Element;
using MKE.ElementFragments;
using MKE.Interface;
using MKE.Point;
using MKE.Utils;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace MKE.Geometry
{
    [JsonConverter(typeof(StringEnumConverter))]
    public enum Surface
    {
        Front, //XZ where y smaller

        Back, //XZ where y bigger

        Top, //XY where z bigger

        Bottom, //XY where z smaller

        Left, //YZ where x smaller

        Right, //YZ where x bigger
    }

    public class GeometryParallelepiped
    {
        public string FunctionForSolutionX { get; set; }
        public string FunctionForSolutionY { get; set; }
        public string FunctionForSolutionZ { get; set; }
        public int StepsT { get; set; }
        [JsonIgnore] public Func<double, double> FuncTransformX { get; set; }
        [JsonIgnore] public Func<double, double> FuncTransformY { get; set; }
        [JsonIgnore] public Func<double, double> FuncTransformZ { get; set; }

        public Dictionary<int, AxisLines> MapXAxisLines { get; set; } = new Dictionary<int, AxisLines>();

        public Dictionary<int, AxisLines> MapYAxisLines { get; set; } = new Dictionary<int, AxisLines>();

        public Dictionary<int, AxisLines> MapZAxisLines { get; set; } = new Dictionary<int, AxisLines>();

        public HashSet<NumberedPoint3D> Points { get; set; } = new HashSet<NumberedPoint3D>(new ComparerPoint()); //в исходной нумерации


        public Dictionary<int, HierarchicalDomain3D<ParallelepipedElement>> MapDomains { get; set; } = new Dictionary<int, HierarchicalDomain3D<ParallelepipedElement>>();

        public List<DirichletCondition> DirichletConditions { get; set; } = new List<DirichletCondition>();

        public List<NeumannCondition> NeumannConditions { get; set; } = new List<NeumannCondition>();


        private static IEnumerator<int> GetSequence()
        {
            var i = 0;

            while (true)
            {
                yield return i;

                i++;
            }
        }

        public Dictionary<Vertex, int> Vertexes = new Dictionary<Vertex, int>(); //исходная нумерация вершин в новую

        public Dictionary<Edge, (int, List<(int, bool)>)> Edges = new Dictionary<Edge, (int, List<(int, bool)>)>(); //ребро в исходной нумерации в номера доп точек на нём

        public Dictionary<SurfaceSquare, (int, List<(int, bool)>)> SurfaceSquares = new Dictionary<SurfaceSquare, (int, List<(int, bool)>)>(); //грань в исходной нумерации в номера доп точек на ней

        public List<int> InnerPoints { get; set; } = new List<int>(); //номера внутренних точек

        private readonly IEnumerator<int> _sequence = GetSequence();

        public int LastSequenceIndex { get; set; }
        public void InitAfterSerialize()
        {
            foreach (var (key, value) in MapXAxisLines)
            {
                value.Initialize();
            }
            foreach (var (key, value) in MapYAxisLines)
            {
                value.Initialize();
            }
            foreach (var (key, value) in MapZAxisLines)
            {
                value.Initialize();
            }
            DirichletConditions.ForEach(x => x.InitFunction());
            NeumannConditions.ForEach(x => x.InitFunction());
            foreach (var (key, value) in MapDomains)
            {
                value.InitializationFunction();
            }
            var parsedFx = DynamicExpressionParser.ParseLambda<DummyT, double>(ParsingConfig.Default, false, FunctionForSolutionX).Compile();
            FuncTransformX = (x) => parsedFx(new DummyT(x));
            var parsedFy = DynamicExpressionParser.ParseLambda<DummyT, double>(ParsingConfig.Default, false, FunctionForSolutionY).Compile();
            FuncTransformY = (x) => parsedFy(new DummyT(x));
            var parsedFz = DynamicExpressionParser.ParseLambda<DummyT, double>(ParsingConfig.Default, false, FunctionForSolutionZ).Compile();
            FuncTransformZ = (x) => parsedFz(new DummyT(x));
        }
        public void GeneratePoints()
        {
            var z = 0;

            foreach (var (key, value) in MapDomains)
            {
                var tempPoints = new Dictionary<(int, int, int), NumberedPoint3D>();

                for (var i = 0; i < MapZAxisLines[value.ZAxisIndex].Axises.Count; i++)
                {
                    var zAxis = MapZAxisLines[value.ZAxisIndex].Axises[i];

                    for (var j = 0; j < MapYAxisLines[value.YAxisIndex].Axises.Count; j++)
                    {
                        var yAxis = MapYAxisLines[value.YAxisIndex].Axises[j];

                        for (var k = 0; k < MapXAxisLines[value.XAxisIndex].Axises.Count; k++)
                        {
                            var xAxis = MapXAxisLines[value.XAxisIndex].Axises[k];
                            var point3D = new NumberedPoint3D(z, xAxis, yAxis, zAxis);

                            if (Points.TryGetValue(point3D, out var existPoint))
                            {
                                tempPoints.Add((k, j, i), existPoint);

                                continue;
                            }

                            tempPoints.Add((k, j, i), point3D);
                            z++;
                        }
                    }
                }

                for (var i = 0; i < MapZAxisLines[value.ZAxisIndex].Axises.Count - 1; i++)
                {
                    for (var j = 0; j < MapYAxisLines[value.YAxisIndex].Axises.Count - 1; j++)
                    {
                        for (var k = 0; k < MapXAxisLines[value.XAxisIndex].Axises.Count - 1; k++)
                        {
                            value.Elements.Add(new ParallelepipedElement(value.Order, new HierarchicalBasisFunction(), new List<NumberedPoint3D>(8)
                                               {
                                                   tempPoints[(k, j, i)], tempPoints[(k + 1, j, i)],
                                                   tempPoints[(k, j + 1, i)], tempPoints[(k + 1, j + 1, i)],
                                                   tempPoints[(k, j, i + 1)], tempPoints[(k + 1, j, i + 1)],
                                                   tempPoints[(k, j + 1, i + 1)], tempPoints[(k + 1, j + 1, i + 1)],
                                               }, 6)
                            { Lambda = value.Lambda, Gamma = value.Gamma }
                                              );
                        }
                    }
                }

                Points.UnionWith(tempPoints.Values);
            }

            Console.WriteLine($"Vertex Count = {Points.Count}\t Element Count = {MapDomains.Values.SelectMany(x => x.Elements).Count()}");
        }
        public void FillConditionsElement()
        {
            var tempPoints = Points.ToDictionary(x => x.Number, y => y);

            foreach (var (surface, addition) in SurfaceSquares)
            {
                var pointA = tempPoints[surface.A.Index];
                var pointB = tempPoints[surface.B.Index];
                var pointC = tempPoints[surface.C.Index];
                var pointD = tempPoints[surface.D.Index];

                var pointO = (pointA + pointD) / 2;

                foreach (var cond in DirichletConditions.Where(cond => cond.Check(pointO, this)))
                {
                    var element = FillElementForCondition(surface, pointA, pointB, pointC, pointD, cond.Surface);
                    cond.Elements.Add(element);
                }
                foreach (var cond in NeumannConditions.Where(cond => cond.Check(pointO, this)))
                {
                    var element = FillElementForCondition(surface, pointA, pointB, pointC, pointD, cond.Surface);
                    cond.Elements.Add(element);
                }
            }
            Console.WriteLine($"Boundary Element Count = {DirichletConditions.Sum(x => x.Elements.Count) + NeumannConditions.Sum(x => x.Elements.Count)}");

        }

        private IElement FillElementForCondition(SurfaceSquare surfaceSquare, NumberedPoint3D pointA, NumberedPoint3D pointB, NumberedPoint3D pointC, NumberedPoint3D pointD, Surface surface)
        {
            var element = new SquaredElement(surfaceSquare.MinOrder, new HierarchicalBasisFunction(), new List<NumberedPoint3D>() { pointA, pointB, pointC, pointD }, 6, surface);

            var fragments = element.TemplateElementInformation;
            var temporaryDictForEdge = new Dictionary<Edge, int>();
            var temporaryDictForSurface = new Dictionary<SurfaceSquare, int>();

            for (int i = 0; i < fragments.Size; i++)
            {
                if (fragments.Vertices.ContainsKey(i))
                {
                    var ver = fragments.Vertices[i];
                    var point = element.Points.ElementAt(ver.Index).Number;
                    var tempVertex = new Vertex(point, element.Order);
                    element.SkipPoint.Add(Vertexes[tempVertex], false);
                    element.LocalToGlobalEnumeration.Add(i, Vertexes[tempVertex]);
                }
                else if (fragments.Edges.ContainsKey(i))
                {
                    var edge = fragments.Edges[i];
                    var A = element.Points.ElementAt(edge.A.Index).Number;
                    var B = element.Points.ElementAt(edge.B.Index).Number;
                    var tempEdge = new Edge(A, B, element.Order);

                    if (temporaryDictForEdge.ContainsKey(tempEdge)) //количество принадлежаших этому ребру дополнительных точек
                    {
                        temporaryDictForEdge[tempEdge] += 1;
                    }
                    else
                    {
                        temporaryDictForEdge.Add(tempEdge, 0);
                    }

                    element.LocalToGlobalEnumeration.Add(i, Edges[tempEdge].Item2[temporaryDictForEdge[tempEdge]].Item1);
                    element.SkipPoint.Add(Edges[tempEdge].Item2[temporaryDictForEdge[tempEdge]].Item1, Edges[tempEdge].Item2[temporaryDictForEdge[tempEdge]].Item2);
                }
                else if (fragments.InnerIndexes.Contains(i))
                {
                    var A = element.Points.ElementAt(0).Number;
                    var B = element.Points.ElementAt(1).Number;
                    var C = element.Points.ElementAt(2).Number;
                    var D = element.Points.ElementAt(3).Number;
                    var tempSurface = new SurfaceSquare(A, B, C, D, element.Order);

                    if (temporaryDictForSurface.ContainsKey(tempSurface))
                    {
                        temporaryDictForSurface[tempSurface] += 1;
                    }
                    else
                    {
                        temporaryDictForSurface.Add(tempSurface, 0);
                    }

                    element.LocalToGlobalEnumeration.Add(i, SurfaceSquares[tempSurface].Item2[temporaryDictForSurface[tempSurface]].Item1);
                    element.SkipPoint.Add(SurfaceSquares[tempSurface].Item2[temporaryDictForSurface[tempSurface]].Item1, SurfaceSquares[tempSurface].Item2[temporaryDictForSurface[tempSurface]].Item2);
                }
            }

            return element;
        }

        public void ExtendsElementWeight()
        {
            _sequence.MoveNext();

            foreach (var (key, value) in MapDomains.OrderBy(x => x.Value.Order))
            {
                foreach (var element in value.Elements)
                {
                    var fragments = element.TemplateElementInformation;
                    var temporaryDictForEdge = new Dictionary<Edge, int>();
                    var temporaryDictForSurface = new Dictionary<SurfaceSquare, int>();
                    TryAddEdgesAndSurfaceForFirstOrder(element, fragments, value.Order);

                    for (int i = 0; i < fragments.Size; i++)
                    {
                        if (fragments.Vertices.ContainsKey(i))
                        {
                            var ver = fragments.Vertices[i];
                            var point = element.Points.ElementAt(ver.Index).Number;
                            var tempVertex = new Vertex(point, value.Order);

                            if (!Vertexes.ContainsKey(tempVertex))
                            {
                                Vertexes.Add(tempVertex, _sequence.Current);
                                _sequence.MoveNext();
                            }

                            element.SkipPoint.Add(Vertexes[tempVertex], false);
                            element.LocalToGlobalEnumeration.Add(i, Vertexes[tempVertex]);
                        }
                        else if (fragments.Edges.ContainsKey(i))
                        {
                            var edge = fragments.Edges[i];
                            var A = element.Points.ElementAt(edge.A.Index).Number;
                            var B = element.Points.ElementAt(edge.B.Index).Number;
                            var tempEdge = new Edge(A, B, value.Order);

                            if (temporaryDictForEdge.ContainsKey(tempEdge)) //количество принадлежаших этому ребру дополнительных точек
                            {
                                temporaryDictForEdge[tempEdge] += 1;
                            }
                            else
                            {
                                temporaryDictForEdge.Add(tempEdge, 0);
                            }

                            if (!Edges.ContainsKey(tempEdge)) //если список всех ребер не содержит его, то добавить
                            {
                                Edges.Add(tempEdge, (value.Order, new List<(int, bool)>()));
                                Edges[tempEdge].Item2.Add((_sequence.Current, false));
                                element.SkipPoint.Add(_sequence.Current, false);
                                element.LocalToGlobalEnumeration.Add(i, _sequence.Current);
                                _sequence.MoveNext();
                            }
                            else
                            {
                                if (Edges[tempEdge].Item2.Count <= temporaryDictForEdge[tempEdge]) //если в списке всех доп точек для данного ребра их меньше, чем в текущем ребре, тогда добавить новые точки
                                {
                                    //если порядок отличается, то все новые точки, должны помечаться как те , которые можно пропустить
                                    Edges[tempEdge].Item2.Add(tempEdge.MinOrder > Edges[tempEdge].Item1 ? (_sequence.Current, true) : (_sequence.Current, false));
                                    element.LocalToGlobalEnumeration.Add(i, _sequence.Current);
                                    _sequence.MoveNext();
                                }
                                else
                                {
                                    element.LocalToGlobalEnumeration.Add(i, Edges[tempEdge].Item2[temporaryDictForEdge[tempEdge]].Item1);
                                }

                                element.SkipPoint.Add(Edges[tempEdge].Item2[temporaryDictForEdge[tempEdge]].Item1, Edges[tempEdge].Item2[temporaryDictForEdge[tempEdge]].Item2);
                            }
                        }
                        else if (fragments.Surfaces.ContainsKey(i))
                        {
                            var indexes = fragments.Surfaces[i].Indexes;
                            var A = element.Points.ElementAt(indexes[0]).Number;
                            var B = element.Points.ElementAt(indexes[1]).Number;
                            var C = element.Points.ElementAt(indexes[2]).Number;
                            var D = element.Points.ElementAt(indexes[3]).Number;
                            var tempSurface = new SurfaceSquare(A, B, C, D, value.Order);

                            if (temporaryDictForSurface.ContainsKey(tempSurface))
                            {
                                temporaryDictForSurface[tempSurface] += 1;
                            }
                            else
                            {
                                temporaryDictForSurface.Add(tempSurface, 0);
                            }

                            if (!SurfaceSquares.ContainsKey(tempSurface))
                            {
                                SurfaceSquares.Add(tempSurface, (value.Order, new List<(int, bool)>()));
                                SurfaceSquares[tempSurface].Item2.Add((_sequence.Current, false));
                                element.SkipPoint.Add(_sequence.Current, false);
                                element.LocalToGlobalEnumeration.Add(i, _sequence.Current);
                                _sequence.MoveNext();
                            }
                            else
                            {
                                if (SurfaceSquares[tempSurface].Item2.Count <= temporaryDictForSurface[tempSurface])
                                {
                                    SurfaceSquares[tempSurface].Item2.Add(tempSurface.MinOrder > SurfaceSquares[tempSurface].Item1
                                                                        ? (_sequence.Current, true)
                                                                        : (_sequence.Current, false));

                                    element.LocalToGlobalEnumeration.Add(i, _sequence.Current);
                                    _sequence.MoveNext();
                                }
                                else
                                {
                                    element.LocalToGlobalEnumeration.Add(i, SurfaceSquares[tempSurface].Item2[temporaryDictForSurface[tempSurface]].Item1);
                                }

                                element.SkipPoint.Add(SurfaceSquares[tempSurface].Item2[temporaryDictForSurface[tempSurface]].Item1, SurfaceSquares[tempSurface].Item2[temporaryDictForSurface[tempSurface]].Item2);
                            }
                        }
                        else if (fragments.InnerIndexes.Contains(i))
                        {
                            InnerPoints.Add(_sequence.Current);
                            element.LocalToGlobalEnumeration.Add(i, _sequence.Current);
                            element.SkipPoint.Add(_sequence.Current, false);
                            _sequence.MoveNext();
                        }
                    }
                }
            }

            LastSequenceIndex = _sequence.Current;
        }

        private void TryAddEdgesAndSurfaceForFirstOrder(IElement element, TemplateElementInformation fragments, int order)
        {
            if (order != 1) return;

            var a = element.Points.ElementAt(fragments.Vertices[0].Index);
            var b = element.Points.ElementAt(fragments.Vertices[1].Index);
            var edge1 = new Edge(a.Number, b.Number, 1);
            if (!Edges.ContainsKey(edge1)) Edges.Add(edge1, (1, new List<(int, bool)>()));

            if (element.Points.Count() < 2)
            {
                return;
            }

            var c = element.Points.ElementAt(fragments.Vertices[2].Index);
            var d = element.Points.ElementAt(fragments.Vertices[3].Index);
            var surface1 = new SurfaceSquare(a.Number, b.Number, c.Number, d.Number, 1);
            var edge2 = new Edge(a.Number, c.Number, 1);
            var edge4 = new Edge(b.Number, d.Number, 1);
            var edge6 = new Edge(c.Number, d.Number, 1);
            if (!Edges.ContainsKey(edge2)) Edges.Add(edge2, (1, new List<(int, bool)>()));
            if (!Edges.ContainsKey(edge4)) Edges.Add(edge4, (1, new List<(int, bool)>()));
            if (!Edges.ContainsKey(edge6)) Edges.Add(edge6, (1, new List<(int, bool)>()));
            if (!SurfaceSquares.ContainsKey(surface1)) SurfaceSquares.Add(surface1, (1, new List<(int, bool)>()));

            if (element.Points.Count() < 4)
            {
                return;
            }

            var a1 = element.Points.ElementAt(fragments.Vertices[4].Index);
            var b1 = element.Points.ElementAt(fragments.Vertices[5].Index);
            var c1 = element.Points.ElementAt(fragments.Vertices[6].Index);
            var d1 = element.Points.ElementAt(fragments.Vertices[7].Index);
            var surface2 = new SurfaceSquare(a1.Number, b1.Number, c1.Number, d1.Number, 1);
            var surface3 = new SurfaceSquare(a.Number, b.Number, a1.Number, b1.Number, 1);
            var surface4 = new SurfaceSquare(c.Number, d.Number, c1.Number, d1.Number, 1);
            var surface5 = new SurfaceSquare(b.Number, d.Number, b1.Number, d1.Number, 1);
            var surface6 = new SurfaceSquare(a.Number, c.Number, a1.Number, c1.Number, 1);
            var edge3 = new Edge(a.Number, a1.Number, 1);
            var edge5 = new Edge(b.Number, b1.Number, 1);
            var edge7 = new Edge(c.Number, c1.Number, 1);
            var edge8 = new Edge(a1.Number, b1.Number, 1);
            var edge9 = new Edge(a1.Number, c1.Number, 1);
            var edge10 = new Edge(b1.Number, d1.Number, 1);
            var edge11 = new Edge(c1.Number, d1.Number, 1);
            var edge12 = new Edge(d.Number, d1.Number, 1);
            if (!Edges.ContainsKey(edge3)) Edges.Add(edge3, (1, new List<(int, bool)>()));
            if (!Edges.ContainsKey(edge5)) Edges.Add(edge5, (1, new List<(int, bool)>()));
            if (!Edges.ContainsKey(edge7)) Edges.Add(edge7, (1, new List<(int, bool)>()));
            if (!Edges.ContainsKey(edge8)) Edges.Add(edge8, (1, new List<(int, bool)>()));
            if (!Edges.ContainsKey(edge9)) Edges.Add(edge9, (1, new List<(int, bool)>()));
            if (!Edges.ContainsKey(edge10)) Edges.Add(edge10, (1, new List<(int, bool)>()));
            if (!Edges.ContainsKey(edge11)) Edges.Add(edge11, (1, new List<(int, bool)>()));
            if (!Edges.ContainsKey(edge12)) Edges.Add(edge12, (1, new List<(int, bool)>()));
            if (!SurfaceSquares.ContainsKey(surface2)) SurfaceSquares.Add(surface2, (1, new List<(int, bool)>()));
            if (!SurfaceSquares.ContainsKey(surface3)) SurfaceSquares.Add(surface3, (1, new List<(int, bool)>()));
            if (!SurfaceSquares.ContainsKey(surface4)) SurfaceSquares.Add(surface4, (1, new List<(int, bool)>()));
            if (!SurfaceSquares.ContainsKey(surface5)) SurfaceSquares.Add(surface5, (1, new List<(int, bool)>()));
            if (!SurfaceSquares.ContainsKey(surface6)) SurfaceSquares.Add(surface6, (1, new List<(int, bool)>()));
        }
    }
}