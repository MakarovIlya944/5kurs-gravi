using System;
using System.Linq.Dynamic.Core;
using MKE.Point;
using Newtonsoft.Json;

namespace MKE.Condition {
    public class DirichletCondition : BaseCondition
    {
        [JsonIgnore] public Func<double, double, double, double> F { get; set; }

        public void InitFunction()
        {
            var parsedF = DynamicExpressionParser.ParseLambda<Point3D, double>(ParsingConfig.Default, false, Function).Compile();
            F = (x, y, z) => parsedF(new Point3D(x, y, z));
        }
    }
}