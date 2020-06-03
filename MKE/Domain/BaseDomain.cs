using System;
using System.Collections.Generic;
using System.Linq.Dynamic.Core;
using System.Linq.Expressions;
using MKE.ElementFragments;
using MKE.Interface;
using MKE.Point;
using Newtonsoft.Json;

namespace MKE.Domain
{
    public class BaseDomain
    {
        public int DomainIndex { get; set; }
        public int XAxisIndex { get; set; }
        public int YAxisIndex { get; set; }
        public int ZAxisIndex { get; set; }
        [JsonIgnore] public List<IElement> Elements { get; set; } = new List<IElement>();
        public string Function { get; set; }
        public string LambdaFunction { get; set; }

        public string GammaFunction { get; set; }
    }

    public class HierarchicalDomain3D<T> : BaseDomain
        where T : IElement
    {
        public int Order { get; set; }
        [JsonIgnore] public GeometryParallelepiped GeometryParallelepiped { get; set; }
        [JsonIgnore] public Func<double, double, double, double> RightFunction { get; set; }
        [JsonIgnore] public Func<double, double, double, double> Lambda { get; set; }
        [JsonIgnore] public Func<double, double, double, double> Gamma { get; set; }

        public void InitFunction()
        {
            var rightParsed = DynamicExpressionParser.ParseLambda<Point3D, double>(ParsingConfig.Default, false, Function).Compile();
            RightFunction = (x, y, z) => rightParsed(new Point3D(x, y, z));
            var lambdaParsed = DynamicExpressionParser.ParseLambda<Point3D, double>(ParsingConfig.Default, false, LambdaFunction).Compile();
            Lambda = (x, y, z) => lambdaParsed(new Point3D(x, y, z));
            var gammaParsed = DynamicExpressionParser.ParseLambda<Point3D, double>(ParsingConfig.Default, false, GammaFunction).Compile();
            Gamma = (x, y, z) => gammaParsed(new Point3D(x, y, z));
        }
    }
}